package main

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/alicebob/miniredis/v2"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/redis/go-redis/v9"
)

type captureSink struct {
	mu     sync.Mutex
	events []Event
}

func (c *captureSink) Publish(e Event) { c.mu.Lock(); c.events = append(c.events, e); c.mu.Unlock() }

func setup(t *testing.T, cfg Config, outcome string) (*Server, *int32, *captureSink, *miniredis.Miniredis) {
	t.Helper()
	var calls int32
	exec := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&calls, 1)
		_, _ = w.Write([]byte(`{"sql":"SELECT 1","outcome":"` + outcome + `","rows":[[1]]}`))
	}))
	t.Cleanup(exec.Close)
	mr := miniredis.RunT(t)
	cfg.ExecutorURL = exec.URL
	if cfg.CacheTTL == 0 {
		cfg.CacheTTL = time.Hour
	}
	sink := &captureSink{}
	s := NewServer(cfg, redis.NewClient(&redis.Options{Addr: mr.Addr()}), sink, prometheus.NewRegistry())
	return s, &calls, sink, mr
}

func ask(s *Server, q string) *httptest.ResponseRecorder {
	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/ask", strings.NewReader(`{"question":"`+q+`","db":"weather"}`))
	req.RemoteAddr = "10.0.0.1:1234"
	s.HandleAsk(rec, req)
	return rec
}

func TestCacheMissThenExactAndNormalizedHit(t *testing.T) {
	s, calls, sink, _ := setup(t, Config{CacheEnabled: true}, "ok")
	if r := ask(s, "How many cities are there?"); r.Header().Get("X-Cache") != "miss" || r.Code != 200 {
		t.Fatalf("first request: cache=%s code=%d", r.Header().Get("X-Cache"), r.Code)
	}
	if r := ask(s, "How many cities are there?"); r.Header().Get("X-Cache") != "hit_exact" {
		t.Fatalf("want hit_exact, got %s", r.Header().Get("X-Cache"))
	}
	if r := ask(s, "  how many CITIES are there  "); r.Header().Get("X-Cache") != "hit_normalized" {
		t.Fatalf("want hit_normalized, got %s", r.Header().Get("X-Cache"))
	}
	if *calls != 1 {
		t.Fatalf("executor called %d times, want 1", *calls)
	}
	if len(sink.events) != 3 || sink.events[1].Cache != "hit_exact" || sink.events[0].Outcome != "ok" {
		t.Fatalf("unexpected events: %+v", sink.events)
	}
}

func TestFailedAnswersAreNotCached(t *testing.T) {
	s, calls, _, _ := setup(t, Config{CacheEnabled: true}, "exec_error")
	ask(s, "broken question")
	ask(s, "broken question")
	if *calls != 2 {
		t.Fatalf("executor called %d times, want 2", *calls)
	}
}

func TestCacheDisabled(t *testing.T) {
	s, calls, _, _ := setup(t, Config{CacheEnabled: false}, "ok")
	ask(s, "same question")
	r := ask(s, "same question")
	if *calls != 2 || r.Header().Get("X-Cache") != "disabled" {
		t.Fatalf("calls=%d cache=%s", *calls, r.Header().Get("X-Cache"))
	}
}

func TestRateLimit(t *testing.T) {
	s, _, _, _ := setup(t, Config{CacheEnabled: true, RateLimit: 2}, "ok")
	ask(s, "q one")
	ask(s, "q two")
	if r := ask(s, "q three"); r.Code != http.StatusTooManyRequests {
		t.Fatalf("want 429, got %d", r.Code)
	}
}

func TestBadRequest(t *testing.T) {
	s, _, _, _ := setup(t, Config{CacheEnabled: true}, "ok")
	rec := httptest.NewRecorder()
	s.HandleAsk(rec, httptest.NewRequest(http.MethodPost, "/ask", strings.NewReader("not json")))
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("want 400, got %d", rec.Code)
	}
}

func TestNormalizeQuestion(t *testing.T) {
	cases := map[string]string{
		"How many cities are there?":              "how many cities are there",
		"  Wie viele  Städte gibt es? ":           "wie viele städte gibt es",
		"Brand#22 or Brand#44, after 2024-03-01.": "brand#22 or brand#44 after 2024-03-01",
	}
	for in, want := range cases {
		if got := NormalizeQuestion(in); got != want {
			t.Errorf("NormalizeQuestion(%q) = %q, want %q", in, got, want)
		}
	}
}
