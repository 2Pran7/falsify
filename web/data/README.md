# web/data

`demo.json` is the only input to the site. It is written by
`scripts/export_demo.py` from Postgres and committed, so Vercel can build
without a database. It is never hand-edited.

`survivorship.json` is written by `scripts/run_survivorship.py --save` and read
by the export. The build fails if `demo.json` is absent: the site never falls
back to sample numbers.
