# Missing-data treatment

This branch (`mmda_enhance`) carries additions that `fix` deliberately does
not. They are not corrections. They change the feature schema on purpose, and
so depart from the published paper.

Read [AUDIT.md](AUDIT.md) first: it explains the defects, and `fix` is where
those are corrected. This branch builds on `fix`.

## What changes

`fix` keeps the published treatment of missing values, because that treatment
is defensible rather than wrong: columns above 70% missing are dropped, the
rest get median imputation fitted on the training split only.

What that treatment assumes is that a gap carries no information. In eICU that
assumption is hard to hold:

- whether a lab was drawn reflects clinical suspicion, so an absent value is
  partly a record of what the team was not worried about;
- the database documents that some hospitals never deployed the interface for
  a given measurement, so an absent value is also partly a property of the
  site rather than the patient;
- a `*_std` feature is the sharpest case. A patient with a single measurement
  has no variability to report. Median imputation gives them a typical one,
  which is an assertion about a patient for whom there is no evidence either
  way.

So this branch keeps the fact of absence as a feature in its own right, and
adds the diagnostic that decides whether doing so was warranted.

### Indicators

`MissingnessIndicators` records, per numeric feature, whether the value was
absent before imputation. A column earns an indicator when its **training**
missing rate falls between `min_rate` and `max_rate`: below the floor there is
nothing to learn from, above the ceiling the indicator is near-constant and the
column is nearly empty anyway.

Which columns qualify is decided on the training split alone, so val and test
cannot add a feature.

```yaml
missingness:
  indicators: true
  min_rate: 0.01
  max_rate: 0.95
  report_by: ["site_id", "mortality"]
```

Set `indicators: false` to get `fix`'s behaviour back.

### The diagnostic

`missingness_report(df, columns, by=[...])` gives the missing rate per feature
overall and per group, and sorts by the spread between groups.

That spread is the part worth reading. A gap that is uniform across hospitals
is a different problem from one that is not: the first is plausibly about the
patients, the second is about the sites, and only the second threatens a
multi-centre model's transfer. Grouping by `mortality` asks the matching
question about the outcome.

AUDIT.md EICU-09 lists both as needed before median imputation can be called
harmless. Neither can be answered without the eICU tables, so this branch
supplies the instrument rather than the answer.

## What this costs

Enabling indicators adds one binary feature per qualifying column, so the
right tower's input grows beyond `fix`'s and no longer corresponds to the
schema the paper describes. `h_cat` and `h_joint` keep their widths (22 and
86): the right tower pools to `d_model` whatever its input width.

That is the whole reason this is a separate branch. Anything that breaks the
published schema belongs here; anything that corrects a defect belongs on
`fix`.

## What is deliberately not here

- **Multiple imputation.** It would give better point estimates, but it
  multiplies the dataset and changes what a "seed" means for the downstream
  model. That is a larger decision than this branch should make quietly.
- **Unit conversion for labs.** `fix` reduces each analyte to its dominant
  unit rather than converting, because conversion factors are per-analyte
  clinical knowledge and inventing them silently would be worse than dropping
  rows visibly.
- **Dropping the 70% filter.** Lowering the threshold interacts with the
  indicators: a column at 90% missing contributes a nearly constant indicator
  and almost no values. Worth testing, once there is data to test it on.

## Status

The indicators and the report are implemented and tested. Their effect on model
performance is unmeasured: that needs the eICU tables and a training run, which
is work for the model repository, not this one.
