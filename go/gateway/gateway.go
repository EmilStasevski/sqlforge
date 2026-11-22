// Package main implements the sqlforge HTTP gateway: Redis cache (exact + normalized question) and
// per-client rate limit in front of the FastAPI executor, Prometheus metrics, and Kafka ask events.
package main

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"strings"
	"time"
	"unicode"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/redis/go-redis/v9"
)

// AskRequest is the public request body.
type AskRequest struct {
	Question string `json:"question"`
	DB       string `json:"db"`
}

// Event is published to Kafka for every request (consumed into ClickHouse).
type Event struct {
	TS        time.Time `json:"ts"`
	Question  string    `json:"question"`
	DB        string    `json:"db"`
	Cache     string    `json:"cache"` // miss | hit_exact | hit_normalized | disabled
	Status    int       `json:"status"`
	Outcome   string    `json:"outcome"`
	SQL       string    `json:"sql"`
	LatencyMS float64   `json:"latency_ms"`
}

// EventSink abstracts the Kafka writer so tests can capture events.
type EventSink interface {
	Publish(e Event)
}

// Config holds runtime settings.
type Config struct {
	ExecutorURL  string
	CacheEnabled bool
	CacheTTL     time.Duration
	RateLimit    int // requests per client per minute; 0 = unlimited
}

// Server is the gateway.
type Server struct {
	cfg    Config
	rdb    *redis.Client
	client *http.Client
	sink   EventSink
	reqs   *prometheus.CounterVec
	lat    *prometheus.HistogramVec
}

// NewServer wires metrics into the given registerer.
func NewServer(cfg Config, rdb *redis.Client, sink EventSink, reg prometheus.Registerer) *Server {
	s := &Server{
		cfg:    cfg,
		rdb:    rdb,
		client: &http.Client{Timeout: 90 * time.Second},
		sink:   sink,
		reqs: prometheus.NewCounterVec(prometheus.CounterOpts{
			Name: "gateway_requests_total", Help: "Requests by cache result and HTTP status.",
		}, []string{"cache", "status"}),
		lat: prometheus.NewHistogramVec(prometheus.HistogramOpts{
			Name:    "gateway_request_seconds",
			Help:    "End-to-end request latency.",
			Buckets: []float64{.001, .0025, .005, .01, .025, .05, .1, .25, .5, 1, 2, 4, 8, 16},
		}, []string{"cache"}),
	}
	reg.MustRegister(s.reqs, s.lat)
	return s
}

// NormalizeQuestion lower-cases, drops sentence punctuation/quotes and collapses whitespace.
// Characters that occur inside literals (#, -, ., :, /) are kept.
func NormalizeQuestion(q string) string {
	var b strings.Builder
	space := false
	for _, r := range strings.ToLower(q) {
		switch {
		case unicode.IsLetter(r) || unicode.IsDigit(r) || strings.ContainsRune("#-.:/_", r):
			if space && b.Len() > 0 {
				b.WriteByte(' ')
			}
			b.WriteRune(r)
			space = false
		default:
			space = true
		}
	}
	return strings.TrimRight(b.String(), ".")
}

func key(prefix, db, q string) string {
	h := sha256.Sum256([]byte(db + "\x00" + q))
	return prefix + hex.EncodeToString(h[:16])
}

func clientIP(r *http.Request) string {
	if xf := r.Header.Get("X-Forwarded-For"); xf != "" {
		return strings.TrimSpace(strings.Split(xf, ",")[0])
	}
	host, _, err := net.SplitHostPort(r.RemoteAddr)
	if err != nil {
		return r.RemoteAddr
	}
	return host
}

// allow implements a fixed-window per-minute rate limit in Redis.
func (s *Server) allow(ctx context.Context, ip string) bool {
	if s.cfg.RateLimit <= 0 {
		return true
	}
	k := fmt.Sprintf("rl:%s:%d", ip, time.Now().Unix()/60)
	n, err := s.rdb.Incr(ctx, k).Result()
	if err != nil {
		return true // fail open: rate limiting must not take the service down
	}
	if n == 1 {
		s.rdb.Expire(ctx, k, 70*time.Second)
	}
	return n <= int64(s.cfg.RateLimit)
}

// HandleAsk serves POST /ask.
func (s *Server) HandleAsk(w http.ResponseWriter, r *http.Request) {
	start := time.Now()
	cache := "disabled"
	status := http.StatusOK
	var body []byte
	var req AskRequest
	defer func() {
		lat := time.Since(start)
		s.reqs.WithLabelValues(cache, fmt.Sprint(status)).Inc()
		s.lat.WithLabelValues(cache).Observe(lat.Seconds())
		var resp struct {
			Outcome string `json:"outcome"`
			SQL     string `json:"sql"`
		}
		_ = json.Unmarshal(body, &resp)
		if s.sink != nil {
			s.sink.Publish(Event{TS: start.UTC(), Question: req.Question, DB: req.DB, Cache: cache, Status: status,
				Outcome: resp.Outcome, SQL: resp.SQL, LatencyMS: float64(lat.Microseconds()) / 1000})
		}
	}()
	fail := func(code int, msg string) {
		status = code
		body = []byte(fmt.Sprintf(`{"error":%q}`, msg))
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(code)
		_, _ = w.Write(body)
	}
	if r.Method != http.MethodPost {
		fail(http.StatusMethodNotAllowed, "POST only")
		return
	}
	ctx := r.Context()
	if !s.allow(ctx, clientIP(r)) {
		fail(http.StatusTooManyRequests, "rate limit exceeded")
		return
	}
	if err := json.NewDecoder(io.LimitReader(r.Body, 1<<16)).Decode(&req); err != nil || len(req.Question) < 3 {
		fail(http.StatusBadRequest, "body must be {\"question\": ..., \"db\": ...}")
		return
	}
	if req.DB == "" {
		req.DB = "weather"
	}
	exactKey := key("q:exact:", req.DB, req.Question)
	normKey := key("q:norm:", req.DB, NormalizeQuestion(req.Question))

	if s.cfg.CacheEnabled {
		cache = "miss"
		if v, err := s.rdb.Get(ctx, exactKey).Bytes(); err == nil {
			cache, body = "hit_exact", v
		} else if v, err := s.rdb.Get(ctx, normKey).Bytes(); err == nil {
			cache, body = "hit_normalized", v
		}
		if cache != "miss" {
			w.Header().Set("Content-Type", "application/json")
			w.Header().Set("X-Cache", cache)
			_, _ = w.Write(body)
			return
		}
	}

	payload, _ := json.Marshal(req)
	resp, err := s.client.Post(s.cfg.ExecutorURL+"/ask", "application/json", bytes.NewReader(payload))
	if err != nil {
		fail(http.StatusBadGateway, "executor unreachable")
		return
	}
	defer resp.Body.Close()
	body, err = io.ReadAll(resp.Body)
	if err != nil {
		fail(http.StatusBadGateway, "executor read error")
		return
	}
	status = resp.StatusCode
	if s.cfg.CacheEnabled && resp.StatusCode == http.StatusOK {
		var parsed struct {
			Outcome string `json:"outcome"`
		}
		if json.Unmarshal(body, &parsed) == nil && parsed.Outcome == "ok" { // only cache successful answers
			pipe := s.rdb.Pipeline()
			pipe.Set(ctx, exactKey, body, s.cfg.CacheTTL)
			pipe.Set(ctx, normKey, body, s.cfg.CacheTTL)
			if _, err := pipe.Exec(ctx); err != nil {
				log.Printf("cache write failed: %v", err)
			}
		}
	}
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("X-Cache", cache)
	w.WriteHeader(status)
	_, _ = w.Write(body)
}
