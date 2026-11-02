# check — 2026-11-02 14:09:03

| passage | result | mean NLL | reference | diff (tol 10%) |
|---|---|---|---|---|
| prose | MATCH | 3.8257 | 3.8313 | 0.15% |
| code | MATCH | 3.3203 | 3.2601 | 1.85% |
| math | MATCH | 4.9174 | 4.9583 | 0.83% |
| json | MATCH | 11.2289 | 10.5476 | 6.46% |
| reasoning | MATCH | 5.2114 | 5.6562 | 7.86% |
| dialogue | MATCH | 6.7964 | 6.8881 | 1.33% |

server `http://10.100.24.2:8000`, model `minimax-m3.1-flash`
