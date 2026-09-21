# Offline MemoryAthena dashboard

Open `index.html` directly in a browser:

```bash
open web/index.html
```

The page is a single static file. It has no CDN dependency, build step, or
network request. Its information architecture follows the useful parts of the
XMemTransfer README—method overview, result snapshot, reproduction map, and
artifact layout—while using MemoryAthena's own data and terminology.

The dashboard is a viewing aid. The authoritative evidence is the raw result
JSON, completion marker, Slurm accounting record, and the paper source listed
in [`../ARTIFACTS.md`](../ARTIFACTS.md).
