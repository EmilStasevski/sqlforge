#!/bin/sh
# Start Ollama and register the fine-tuned GGUF (runs/serve/sqlforge-q4_k_m.gguf) as model "sqlforge".
ollama serve &
pid=$!
until ollama list >/dev/null 2>&1; do sleep 1; done
if ! ollama list | grep -q '^sqlforge'; then
  if [ -f /models/Modelfile ]; then
    (cd /models && ollama create sqlforge -f Modelfile)
  else
    echo "no /models/Modelfile yet (run: make gguf)"
  fi
fi
wait $pid
