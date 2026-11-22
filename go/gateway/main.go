package main

import (
	"context"
	"encoding/json"
	"log"
	"net/http"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"github.com/redis/go-redis/v9"
	"github.com/segmentio/kafka-go"
)

// KafkaSink publishes events asynchronously; a full buffer drops events instead of blocking requests.
type KafkaSink struct {
	ch chan Event
	w  *kafka.Writer
}

// NewKafkaSink starts the background writer.
func NewKafkaSink(brokers []string, topic string) *KafkaSink {
	s := &KafkaSink{
		ch: make(chan Event, 10000),
		w: &kafka.Writer{
			Addr:                   kafka.TCP(brokers...),
			Topic:                  topic,
			Balancer:               &kafka.LeastBytes{},
			BatchTimeout:           50 * time.Millisecond,
			AllowAutoTopicCreation: true,
		},
	}
	go s.loop()
	return s
}

func (s *KafkaSink) loop() {
	for e := range s.ch {
		b, _ := json.Marshal(e)
		if err := s.w.WriteMessages(context.Background(), kafka.Message{Value: b}); err != nil {
			log.Printf("kafka write failed: %v", err)
		}
	}
}

// Publish enqueues without blocking.
func (s *KafkaSink) Publish(e Event) {
	select {
	case s.ch <- e:
	default:
	}
}

func env(k, def string) string {
	if v := os.Getenv(k); v != "" {
		return v
	}
	return def
}

func main() {
	ttl, _ := time.ParseDuration(env("CACHE_TTL", "1h"))
	rl, _ := strconv.Atoi(env("RATE_LIMIT_PER_MIN", "0"))
	cfg := Config{
		ExecutorURL:  env("EXECUTOR_URL", "http://localhost:8000"),
		CacheEnabled: env("CACHE_ENABLED", "true") == "true",
		CacheTTL:     ttl,
		RateLimit:    rl,
	}
	rdb := redis.NewClient(&redis.Options{Addr: env("REDIS_ADDR", "localhost:6379")})
	var sink EventSink
	if brokers := env("KAFKA_BROKERS", ""); brokers != "" {
		sink = NewKafkaSink(strings.Split(brokers, ","), env("KAFKA_TOPIC", "ask.events"))
	}
	s := NewServer(cfg, rdb, sink, prometheus.DefaultRegisterer)
	mux := http.NewServeMux()
	mux.HandleFunc("/ask", s.HandleAsk)
	mux.Handle("/metrics", promhttp.Handler())
	mux.HandleFunc("/health", func(w http.ResponseWriter, _ *http.Request) { _, _ = w.Write([]byte("ok")) })
	addr := env("ADDR", ":8080")
	log.Printf("gateway on %s (cache=%v, executor=%s)", addr, cfg.CacheEnabled, cfg.ExecutorURL)
	srv := &http.Server{Addr: addr, Handler: mux, ReadHeaderTimeout: 5 * time.Second}
	log.Fatal(srv.ListenAndServe())
}
