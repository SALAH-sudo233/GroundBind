# Historical recovery

The complete pre-cleanup Git tree is commit
`03eadf015b335a6f17238e2470d81776259524c4`.

```sh
git show 03eadf015b335a6f17238e2470d81776259524c4:README.md
git archive 03eadf015b335a6f17238e2470d81776259524c4 -o vsight-before-v051.tar
```

No Git history was rewritten. The local migration also contains a verified Git
bundle `V-SIGHT-before-v051-cleanup.bundle` and a complete old working-tree copy
under `archives/V-SIGHT-before-v051-20260929`.

The active tree no longer ships early E1/E2/E3 candidate-pool experiments, TRACE,
CCV/CABLE, agentic-drift/flywheel pipelines, their stale directions and duplicate
results as current work. They remain recoverable from the above commit and local
archive. Old asset snapshots also remain intact in the local history entry.

The retained method includes the reported baseline and single-arm ablations;
keeping only favorable result rows would make the paper less reproducible.
