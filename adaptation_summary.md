# Adaptation modelling: how it works

## What this adds

The base pipeline (rules 00–17) builds one flood map per basin × scenario. This
extension lets you take an already-built baseline scenario run and ask "what if
we applied adaptation measure X here?" — for a chosen strategy (a named bundle of
measures at specific parameter values), in two different ways that can be
compared side by side:

- **pre**: apply the measures to the SFINCS model itself, then rerun the flood
  simulation from scratch. Physically correct, but as expensive as a normal
  model run.
- **post**: apply the measures directly to the baseline's already-computed flood
  depth map, by editing pixels. Cheap and fast, but an approximation — no
  physics are recomputed, so effects like backwater or redirected flow are
  never captured.

Both methods start from the same baseline scenario run and produce the same
kind of output (a flood depth raster + a metrics CSV), so results are directly
comparable. Nothing about the existing baseline pipeline changes — adaptation
is entirely opt-in and only runs when explicitly requested.

## Opting in: strategies and methods

Two new selectors sit alongside the existing basin/scenario selectors:

- **Strategy** — which named bundle of measures to apply (defined in
  `config/adaptation_strategies.yml`, e.g. "grey protect-open" = a coastal +
  river levee at 2 m). Unlike scenarios, there's no default strategy — you
  always have to name one explicitly on the command line, since running
  adaptation unprompted would silently multiply every build by however many
  strategies exist.
- **Method** — `pre`, `post`, or both (both is the default once you've picked a
  strategy). This determines which of the two chains described below actually
  runs.

Each strategy is a list of measures applied in a fixed order (as written in the
YAML), each measure type carrying its own parameter values (e.g. levee height,
pump discharge). The catalogue of what measure *types* exist and what
parameters they take — independent of any specific strategy — lives separately
in `config/measures.yml`.

## The attribution mask (what makes `post` possible)

To let the `post` method edit "just the coastal flooding" or "just the river
flooding" out of a raster, it needs to know, pixel by pixel, *why* that pixel
is flooded. That's what the attribution mask provides: a per-basin×scenario
classification raster with four classes — river-dominated, coastal-dominated,
compound (both), and permanent water.

It's built without running any new SFINCS simulations, by comparing three flood
maps that (mostly) already exist:
- the target scenario's own **river-only** counterpart run (same river return
  period, no surge),
- its own **coastal-only** counterpart run (same surge return period, no
  river),
- the basin's **spin-up** run (a fixed, low return-period run shared by every
  scenario), whose wet cells represent permanent water/baseline conditions
  rather than event-driven flooding — this class overrides the other three,
  since a perennial river channel shouldn't be attributed to "river flooding"
  from the event itself.

A pixel flooded in the river-only run but not the coastal-only run is
river-dominated; flooded in both is compound; flooded in neither but wet in the
spin-up is permanent water; and so on. The counterpart runs are matched
automatically by return period — a lookup function inspects the target
scenario's own river/surge return periods and finds the sibling single-driver
scenarios with matching values. This only works for scenarios that actually
have such a sibling defined in `config/scenarios.yml`; scenarios without one
raise a clear error rather than silently attributing against a mismatched
return period.

This mask is computed once per basin × scenario, shared across every strategy
applied to that scenario (not recomputed per strategy), and only the `post`
method ever touches it — `pre` reruns SFINCS directly and has no use for it.

## The `pre` chain

This reuses the existing baseline pipeline's own machinery almost entirely
unmodified — the same scripts that build the SFINCS forcing, run the event,
and compute flood metrics for a normal scenario run are reused as-is here, just
pointed at new output locations. Only one genuinely new step is added at the
front:

1. **Apply measures to the model.** Starting from a read-only copy of the
   basin's SFINCS skeleton (grid, elevation, mask, etc.), each measure in the
   strategy is applied in order directly to the model — e.g. a levee measure
   adds a weir at the given height and location, a retreat measure changes
   land-use classification and roughness within an eligible flooded area. Only
   the parts of the model a strategy's measures actually touch get rewritten;
   everything untouched is carried forward from the skeleton by reference,
   exactly like the baseline build already does. If the strategy includes a
   retreat measure, that step also needs the scenario's own already-computed
   baseline flood map, since which cells are eligible for retreat depends on
   how deep and where flooding already occurs at this specific return period —
   this is why the apply step is scoped per basin × scenario × strategy, not
   just per basin × strategy.
2. **Rebuild forcing.** Same forcing-construction step the baseline pipeline
   already uses, pointed at the adapted model instead of the untouched one.
3. **Run the event.** Same SFINCS-execution step, reused unmodified.
4. **Compute metrics.** Same flood-metrics computation, reused unmodified,
   producing a flood depth raster and a metrics CSV in the strategy's own
   output folder.

The basin-level spin-up (the shared initial condition every scenario's event
run starts from) is never rerun per strategy — none of the currently
implemented measures change the coarse grid's elevation or active area, so the
existing spin-up restart remains valid regardless of which adaptation measures
get applied downstream.

## The `post` chain

This is a single step that never touches SFINCS: it starts from the baseline
scenario's already-computed flood depth raster and applies each measure in the
strategy in turn, each one reading the *previous* measure's output raster and
writing a new one — so effects compound across a strategy with multiple
measures, in the order they're listed.

Each measure type has its own rule for how it edits the raster, generally
using the attribution mask to restrict the edit to the flood source(s) it
actually addresses:

- A **coastal barrier or levee** compares its crest height against the peak
  coastal water level from the scenario's own model output; if the crest wins,
  every coastal and compound pixel is cleared, otherwise nothing changes — an
  all-or-nothing threshold check per structure, not a partial-protection curve.
- A **combined coastal + river levee** checks both crest heights against their
  respective peak water levels independently, clearing river-only pixels if
  the river levee holds, coastal-only pixels if the coastal levee holds, and
  compound pixels only if *both* hold (since either source alone could still
  cause compound flooding).
- A **dike ring** works the same threshold-vs-water-level logic, but restricted
  spatially to whatever area a supplied ring polygon encloses, rather than
  restricted by flood-source class.
- **Land reclamation** (nature-based coastline extension) reduces coastal and
  compound pixel depths by a flat amount derived from how far the coastline is
  pushed out and an attenuation rate, rather than fully removing flooding.
- **Water retention and pumps** both convert a physical capacity (a storage
  fraction of a fixed baseline excess volume, or a pump discharge rate over the
  event duration) into an equivalent uniform depth reduction spread across the
  river/compound pixels, clipped so depth never goes negative — a graded
  reduction rather than a binary on/off.
- **Retreat** is the odd one out: it never touches the hazard raster at all
  (copied through unchanged). Instead it reclassifies the deepest-flooded
  share of eligible urban, flooded cells to a different land-use class in a
  new land-use raster, which is what feeds into the final metrics computation
  instead of the basin's raw land-use — the "adaptation" here is exposure
  reduction (fewer urban cells counted as at risk), not hazard reduction.

After the chain finishes, metrics are computed once on the final raster (and
final land-use raster, if retreat ran), using the same metrics logic the
baseline pipeline uses elsewhere, so pre/post/baseline metrics are all
comparable on equal terms.

## Output layout

Every adaptation output — for both methods — lands under each scenario's own
results folder, in an `adaptation/<pre or post>/<strategy>/` subfolder, flat
(no separate visuals/metrics split like the baseline pipeline uses), containing
the final flood depth raster, the metrics CSV, and a couple of diagnostic
plots. For `pre`, the SFINCS model's own working files (its input deck, the
rebuilt forcing) live in their own nested subfolders inside that same strategy
folder, since SFINCS needs specific filenames in specific working directories
to run at all — everything that isn't part of the model's internal working
files sits in the flat top level alongside the other method's own outputs.

## Comparing results

A final aggregation step collects every baseline, pre-, and post-method metrics
CSV currently in scope for a basin (across every scenario and strategy
selected on the command line) into one combined table, labelling each row by
method and strategy so the three can be filtered, grouped, and compared
directly — e.g. to see how far the post method's approximation diverges from
the pre method's physically rerun result for the same strategy and scenario.
