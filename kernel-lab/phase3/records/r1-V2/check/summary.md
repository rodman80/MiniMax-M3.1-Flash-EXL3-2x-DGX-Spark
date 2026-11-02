# check — 2026-11-02 14:21:56

| passage | result | mean NLL | reference | diff (tol 10%) |
|---|---|---|---|---|
| prose | MATCH | 3.8257 | 3.8313 | 0.15% |
| code | MATCH | 3.3570 | 3.2601 | 2.97% |
| math | MATCH | 4.9349 | 4.9583 | 0.47% |
| json | MATCH | 11.2158 | 10.5476 | 6.34% |
| reasoning | MATCH | 5.2114 | 5.6562 | 7.86% |
| dialogue | MATCH | 7.2621 | 6.8881 | 5.43% |

server `http://10.100.24.2:8000`, model `minimax-m3.1-flash`
