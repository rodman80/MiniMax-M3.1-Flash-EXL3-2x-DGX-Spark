# check — 2026-11-02 14:33:00

| passage | result | mean NLL | reference | diff (tol 10%) |
|---|---|---|---|---|
| prose | MATCH | 3.7821 | 3.8313 | 1.28% |
| code | MATCH | 3.3314 | 3.2601 | 2.19% |
| math | MATCH | 4.9363 | 4.9583 | 0.44% |
| json | MATCH | 11.1738 | 10.5476 | 5.94% |
| reasoning | MATCH | 5.2039 | 5.6562 | 8.00% |
| dialogue | MATCH | 7.2621 | 6.8881 | 5.43% |

server `http://10.100.24.2:8000`, model `minimax-m3.1-flash`
