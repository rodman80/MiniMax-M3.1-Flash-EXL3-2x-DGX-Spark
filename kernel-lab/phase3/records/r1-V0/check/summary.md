# check — 2026-11-02 13:58:08

| passage | result | mean NLL | reference | diff (tol 10%) |
|---|---|---|---|---|
| prose | MATCH | 3.7961 | 3.8313 | 0.92% |
| code | MATCH | 3.3314 | 3.2601 | 2.19% |
| math | MATCH | 4.9349 | 4.9583 | 0.47% |
| json | MATCH | 11.1387 | 10.5476 | 5.60% |
| reasoning | MATCH | 5.2114 | 5.6562 | 7.86% |
| dialogue | MATCH | 6.7947 | 6.8881 | 1.36% |

server `http://10.100.24.2:8000`, model `minimax-m3.1-flash`
