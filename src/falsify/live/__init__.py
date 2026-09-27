"""Live runs: a visitor types a hypothesis on the public site and the agent runs it.

limits   the rules that decide whether a run may start, pure and tested
store    live_run rows: a separate table, never research_note
worker   one run at a time, off the request thread
app      the FastAPI surface: submit, poll, and a token-guarded admin list

THE ONE RULE THAT SHAPES ALL OF IT: a visitor's run is shown to that visitor and
to the owner, and never to anyone else. It is stored in `live_run`, not
`research_note`, so `scripts/export_demo.py` cannot pick it up and put a
stranger's text on the public page. Publishing one is a deliberate act by the
owner, never a side effect of someone typing.
"""
