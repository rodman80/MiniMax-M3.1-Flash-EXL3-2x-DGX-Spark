# check — 2026-11-01 21:44:08

| passage | result | mean NLL | reference | diff (tol 10%) |
|---|---|---|---|---|
| prose | MATCH | 3.7895 | 3.8313 | 1.09% |
| code | MATCH | 3.3314 | 3.2601 | 2.19% |
| math | MATCH | 4.9363 | 4.9583 | 0.44% |
| json | MATCH | 11.1738 | 10.5476 | 5.94% |
| reasoning | MATCH | 5.2215 | 5.6562 | 7.69% |
| dialogue | MATCH | 6.7947 | 6.8881 | 1.36% |

server `http://10.100.24.2:8000`, model `minimax-m3.1-flash`
