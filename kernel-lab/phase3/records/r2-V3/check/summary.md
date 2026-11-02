# check — 2026-11-02 14:40:24

| passage | result | mean NLL | reference | diff (tol 10%) |
|---|---|---|---|---|
| prose | MATCH | 3.8103 | 3.8313 | 0.55% |
| code | MATCH | 3.3131 | 3.2601 | 1.63% |
| math | MATCH | 4.9363 | 4.9583 | 0.44% |
| json | MATCH | 11.1387 | 10.5476 | 5.60% |
| reasoning | MATCH | 5.2039 | 5.6562 | 8.00% |
| dialogue | MATCH | 6.7964 | 6.8881 | 1.33% |

server `http://10.100.24.2:8000`, model `minimax-m3.1-flash`
