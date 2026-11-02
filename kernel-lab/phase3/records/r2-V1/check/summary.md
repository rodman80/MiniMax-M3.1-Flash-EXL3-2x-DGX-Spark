# check — 2026-11-02 15:00:36

| passage | result | mean NLL | reference | diff (tol 10%) |
|---|---|---|---|---|
| prose | MATCH | 3.8082 | 3.8313 | 0.60% |
| code | MATCH | 3.4094 | 3.2601 | 4.58% |
| math | MATCH | 4.9174 | 4.9583 | 0.83% |
| json | MATCH | 11.4707 | 10.5476 | 8.75% |
| reasoning | MATCH | 5.2114 | 5.6562 | 7.86% |
| dialogue | MATCH | 6.7947 | 6.8881 | 1.36% |

server `http://10.100.24.2:8000`, model `minimax-m3.1-flash`
