# TODO

Decision table; mark each row's decision. Previous table: git show 580633b:TODO.md

## Behaviour

| id | file:line | issue (one line) | kind | effort | impact | recommendation |
|---|---|---|---|---|---|---|
| B22 | workflow/src/surge.py:635 | rule select_surge_stations prints shapely 'invalid value encountered in distance' for every basin; COAST-RP coordinates have no NaN, so a NaN/empty geometry on the domain side is the likely source; stations are still found (Ebro: 9 within 10 km) | behaviour | S | none seen | decide: find the NaN geometry and drop it explicitly |

## Structure

| id | file:line | issue (one line) | kind | effort | impact | recommendation |
|---|---|---|---|---|---|---|

## Text

| id | file:line | issue (one line) | kind | effort | impact | recommendation |
|---|---|---|---|---|---|---|
| X57 | LICENSE; CITATION.cff | missing before sharing | text | S | repo shareable and citable | decide: licence; then fix |
