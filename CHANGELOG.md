## 0.104.0 (2025-02-14)

### Feat

- warmup updates

## 0.103.1 (2025-02-14)

### Fix

- remove gamma interp from epcor scrubber cfg
- increase lso initial lr
- respect tag config aggregation in offline chunk loading

## 0.103.0 (2025-02-14)

### Feat

- remove last remaining concept of interpolation

### Refactor

- remove interpolation from monte carlo eval
- no longer use interpolation for environment seed
- remove the need to specify a transition creator

## 0.102.0 (2025-02-14)

### Feat

- add filter for pre dp or action change

### Fix

- add action change to step batch
- decouple action change from decision point

### Refactor

- countdown uses 0 instead of steps_per_decision on decision point
- remove tag config from transition creator
- ignore step timestamp in step iterator

### Perf

- copy instead of deepcopy in all-the-time transition creator

## 0.101.0 (2025-02-14)

### Feat

- consider out-of-bounds delta actions as nan instead of clipping
- broaden clip xform to do general bounds checking

## 0.100.0 (2025-02-14)

### Feat

- action variance and q val metrics

## 0.99.0 (2025-02-14)

### Feat

- logging backtrack steps and step_size

### Fix

- lso example
- pass optimizer lr to LSO

## 0.98.0 (2025-02-13)

### Feat

- add timestamp to steps

### Fix

- ensure None values are ignored in replay buffer
- ignore step timestamp in buffer

## 0.97.0 (2025-02-13)

### Feat

- add ability to override directly loaded config options
- delta action cfg encantation is computed from feature flag
- allow GAC to have different delta percents per setpoint tag

### Fix

- configs should no longer be assumed frozen

## 0.96.0 (2025-02-13)

### Feat

- support specifying a subset of tags in sim environments

### Refactor

- cleanup lingering default values in cfg YAMLs
- pull preprocessor normalization defaults into computed cfgs
- move oddity detection defaults into global default

## 0.95.0 (2025-02-13)

### Feat

- Web Client has can verify if DB and OPC connections are available (#506)

## 0.94.0 (2025-02-13)

### Feat

- allow specifying a single top-level default imputer for all tags

### Fix

- ignore checkpointing the optimizer
- annotate agent config types with discriminator

### Refactor

- default out stepsizes and action samples in cfgs
- default out several agent configs

## 0.93.3 (2025-02-13)

### Refactor

- remove NStepInfo dataclass
- simplify all-the-time TC

## 0.93.2 (2025-02-13)

### Fix

- update dependency versions

### Refactor

- remove all buffer config overrides in favor of defaults
- remove lots of unnecessary is_meta and sc: null cfgs
- prefer global sc defaults over tag-specific sc xforms
- remove normalizer from default sc cfgs
- remove many unused cfg values

## 0.93.1 (2025-02-13)

### Fix

- more debugging
- Small fix for Docker build

## 0.93.0 (2025-02-13)

### Feat

- mpc on saturation

## 0.92.0 (2025-02-12)

### Feat

- transition_len metric

### Fix

- set app state when loading buffer in gac
- ignore app state in pickle

## 0.91.0 (2025-02-12)

### Feat

- metric logging loss on n_most_recent samples

### Fix

- ingress will work iwth n_most_recent > 1

## 0.90.0 (2025-02-12)

### Feat

- write buffer size metric

### Fix

- pipe app state to buffers in aux agents

## 0.89.0 (2025-02-12)

### Feat

- ingress loss metrics
- genearlized combined replay

### Fix

- check if idx empty
- adding da filtering after sample

### Refactor

- policy loss

## 0.88.1 (2025-02-11)

### Fix

- gac test var names
- remove unused return values from _update_policy
- delta_action_bug

## 0.88.0 (2025-02-11)

### Feat

- wire Actor-Critic eval and evals plotting into online interaction

### Refactor

- access MainConfig through app_state

## 0.87.0 (2025-02-11)

### Feat

- eval_batch cfg in GAC

## 0.86.1 (2025-02-11)

### Fix

- include timestamp in opc write
- always take action if requested from event bus

## 0.86.0 (2025-02-11)

### Feat

- eval batch for lso with critic + refactor of critic loss to return single tensor
- separate eval batch for policy

### Fix

- remove extra line in saturation test
- loss is not list (again)
- actually avg_critic_loss
- avg_critic_loss in tests
- add 0 to traces in saturation behaviour
- no gradients in closures

## 0.85.2 (2025-02-11)

### Fix

- add default seed to the ensemble buffer

## 0.85.1 (2025-02-11)

### Fix

- migrate new configs to new schema
- move db infra configs into dedicated top-level config

### Refactor

- pull action_tolerance into interaction instead of env

## 0.85.0 (2025-02-11)

### Feat

- Epcor Solar dataloader adapted to s3

## 0.84.1 (2025-02-10)

### Fix

- style
- fix input size

### Refactor

- refactor ensemble critic

## 0.84.0 (2025-02-10)

### Feat

- multiple action saturation

## 0.83.0 (2025-02-10)

### Feat

- add application uri to opc config

### Fix

- increase size of offline chunk load in epcor scrubber cfg
- add countdown to epcor scrubber cfg
- move action/obs period from env to interaction in scrubber cfg
- remove redundant obs/action period from env cfg in dep mountain car continuous
- make chkpoint dir name windows compatible
- update scrubber config
- explicitly use timestamp with time zone in sql queries/writes

## 0.82.0 (2025-02-08)

### Feat

- Actor-Critic plotting code + evals plotting tooling

### Refactor

- update test to reflect changes made to the jsonb object written to TSDB by Actor-Critic eval

## 0.81.0 (2025-02-07)

### Feat

- added remaining catalyst-ui foundational components
- added reusable setup main config nav buttons
- added duration component and sample usage
- added steps/progress-bar component, refactor for inf. rerender
- added heading component
- added fieldset and input
- port to local forage, fixed lint and build errors, foundational UI components
- initial yaml creation functionality

### Fix

- addressed form upload button for initial file upload
- addressed post-rebase pytest failures, removed stub_required route, uv.lock version
- updated minimal config
- clear  message when we click Clear Setup Config

## 0.80.0 (2025-02-07)

### Feat

- critics use linesearch by default

### Refactor

- remove defaulted hypers from test configs

## 0.79.0 (2025-02-07)

### Feat

- implemented Monte-Carlo Eval plotting function and a function that calls all metrics plotting functions

### Refactor

- addressing Andy's save_path feedback
- updated offline utils tests to reflect changes made to configs and OfflineTraining class
- Enforce all evaluators to be executed at the same offline/online training iters

## 0.78.1 (2025-02-07)

### Fix

- ignore type errors in three_tanks

## 0.78.0 (2025-02-07)

### Feat

- pre and post pipeline hooks

### Fix

- rm entropy and n_samplre from offline_pretrianing.yaml
- slight modifications to behaviour tests to make stuff pass
- rm n_sampler_updates from configs
- ensemble defaults

### Refactor

- pulled out percentile ranking logic
- further unification of sampler and proposal updates
- nearly unified sampler and actor updates
- update sampler
- critic loss
- remove awful update info tuple

## 0.77.0 (2025-02-07)

### BREAKING CHANGE

- a result of these simplifications is that configs
that manipulated the timing controls within the env will no longer
be valid.
- this adds a new responsibility to the interaction cfg
and that is not backwards compatible or defaultable. In fact, the new
definition of the interaction config should be used to define defaults
in the other places where `obs_period` and `action_period` are being
used.

### Feat

- remove env.obs_period interpolations throughout configs
- add computed and sanitizer config utilities

### Fix

- rename eval->eval_cfgs in vww config
- migrate pipeline_batch_duration to timedelta

### Refactor

- make action tolerance a computed config
- pull env cfgs into single source
- make interaction cfg responsible for periods

## 0.76.0 (2025-02-06)

### Feat

- added autoencoder coverage function
- initial coverage framework: incld kde and neighbours + tests

## 0.75.2 (2025-02-06)

### Fix

- style issues.
- greedyAC on gpu and different batch for linesearch

## 0.75.1 (2025-02-06)

### Fix

- remove linesearch dependencies in dockerfile build

## 0.75.0 (2025-02-06)

### Feat

- configure historical data initial timestamp

### Fix

- enable int node id in tag cfg

## 0.74.0 (2025-02-06)

### Feat

- wire up lso
- reintroduce lso optimizer monorepo style

### Fix

- fix imports in test_sls
- fix (suppress) linting messages in lso

## 0.73.0 (2025-02-06)

### Feat

- normalize countdown features

### Fix

- countdown feature supports delta actions

## 0.72.0 (2025-02-06)

### Feat

- add per tag aggregation to tagconfig

### Fix

- fix yaml dumper to support strenums

## 0.71.0 (2025-02-06)

### Feat

- support OPC security policy Basic256Sha256 SignAndEncrypt

## 0.70.1 (2025-02-04)

### Fix

- Deployment Async Env now performs VariantType aware writes to OPC (#476)

## 0.70.0 (2025-02-03)

### Feat

- Eval Table read()

### Fix

- use unique name for eval table idx
- EvalsTable read() needed to match read() in EvalTableProtocol

## 0.69.2 (2025-02-03)

### Fix

- remove a BUG in greedy_ac

## 0.69.1 (2025-01-31)

### Fix

- let ensemble critic use armijo adam

## 0.69.0 (2025-01-31)

### Feat

- implemented Metrics Table read() method for both TSDB and Pandas

### Refactor

- Added blocking_sync() before read() and updated MetricsTable fixture in tests

## 0.68.0 (2025-01-31)

### Feat

- enable delta actions in deployment interaction
- make ac action labelling aware of delta actions
- add ability to consume and produce delta actions in GAC
- add clip xform
- add is_delta_transformed static method to Delta xform

### Fix

- remove double delta in greedy_ac update
- move action_dim modification due to integration bug from stale PR
- wire delta bounds through agent_cfg to gac
- increase magic level to decrease collision chance Δ
- define initial action in gym envs as 0

### Refactor

- expose tags used by constructor as protected attribute

## 0.67.0 (2025-01-31)

### Feat

- added saturation to the behaviour test suite
- add state and action logging to sim_interaction

### Fix

- accidentally deleted evaluation in previous commit

## 0.66.0 (2025-01-31)

### Feat

- support agent step with msg bus
- dep async env writes to agent step OPC node
- opc client can optionally sync env steps to agent step
- option to disable loading checkpoints

### Fix

- update dep smoke test to use new opc sim config
- heartbeat dtype

### Refactor

- gym sim configs
- log rewards fn in dep interaction

## 0.65.0 (2025-01-30)

### Feat

- add state and action logging to sim_interaction

### Fix

- accidentally deleted evaluation in previous commit

## 0.64.1 (2025-01-30)

### Fix

- ruff
- ruff
- remove network_linesearch and refactor linesearch adam

## 0.64.0 (2025-01-30)

### Feat

- add ensemble buffer to reweight historical data
- replay buffers take data_mode as feed argument

### Fix

- prefer discriminated union pattern for config schemas

## 0.63.0 (2025-01-30)

### Feat

- add mc evaluator to online mode

### Fix

- fix integration bug due to stale PRs
- add missing type hints
- clear numpy deprecation warning
- only grab the rewards column from rewards df

### Refactor

- add main config to app_state
- generalize iteration counter label
- remove nullables and simplify mc eval implementation
- move transient pipe_return object off of class attributes
- remove conditional init from monte_carlo eval

## 0.62.1 (2025-01-30)

### Fix

- mc eval config is not a base eval config
- prefer Field to field in config dataclasses
- have mc eval own its own config parameters
- default resolution with Fields now walks through default_factory schemas

## 0.62.0 (2025-01-30)

### Feat

- Implemented Actor-Critic evaluator

## 0.61.0 (2025-01-30)

### Feat

- implemented EvalWriter

### Refactor

- reverted name 'metrics_writer' back to 'metrics' and renamed 'eval_writer' to 'evals'. In MainConfig, renamed 'evals' to 'eval_cfgs'
- changed 'metrics' to 'metrics_writer' in test_main.py
- changed 'metrics' to 'metrics_writer' in interaction
- renaming app_state's 'metrics' attribute to 'metrics_writer'
- rename eval/writer.py to eval/metrics_writer.py

## 0.60.0 (2025-01-30)

### Feat

- Added a FastAPI endpoint for searching OPC variables (#461)

## 0.59.0 (2025-01-29)

### Feat

- change log_files bool in main cfg to log_path Path

## 0.58.3 (2025-01-29)

### Fix

- comparator to_numpy converts to float dtype

## 0.58.2 (2025-01-29)

### Fix

- fix style
- fix linesearch initlization

## 0.58.1 (2025-01-29)

### Fix

- use table name to uniquely identify their respective idx

### Perf

- set default low watermark to 1 for metrics

## 0.58.0 (2025-01-29)

### Feat

- support different aggregation method for different tags

### Fix

- fix style issue
- fixed batch and single read header
- make ruff happy
- fix pyright
- provide fallback aggregation method
- add back debug message
- add back debug message
- allow NaN and to use tryconnectcontextmanager
- add back missing columns
- fix styles

## 0.57.0 (2025-01-29)

### Feat

- inverse xform

## 0.56.0 (2025-01-28)

### Feat

- implemented Monte-Carlo evaluator as an exclusively offline training evaluator

### Fix

- PipelineReturn needed __iadd__ instead of __add__ to update the object in place. Also needed to return the object

### Refactor

- moved assertion into helper method
- moved actor, v_critic, and q_critic initialization to BaseAC for typing purposes
- ensure every algorithm that inherits from BaseAC has 'actor', 'q_critic', and 'v_critic' attributes

## 0.55.2 (2025-01-28)

### Fix

- update pilot cf operating range
- default heartbeat to None instead of relying on enabled flag
- log reward in dep interaction
- update dv pilot config
- decouple tag name from node name in dep env action nodes

## 0.55.1 (2025-01-28)

### Fix

- update tests to handle norm default from_data=False
- do not normalize meta tags in preprocess stage
- from_data defaults to false in normalizer

## 0.55.0 (2025-01-28)

### Feat

- support for OpenAPI generated TypeScript api client
- added routing and state/query library, stricter eslint rules

## 0.54.0 (2025-01-28)

### Feat

- config validation endpoint
- Creating configs from dict in server

### Fix

- first pass at pr comments
- respond to PR comments
- added json and yaml requests

## 0.53.0 (2025-01-27)

### Feat

- replace xform

## 0.52.3 (2025-01-27)

### Fix

- create outputs dir if it does not exist

## 0.52.2 (2025-01-27)

### Fix

- ensure state constructor columns are always sorted
- ensure action constructor columns are always sorted

## 0.52.1 (2025-01-27)

### Fix

- ensure default output_dir is a Path object

## 0.52.0 (2025-01-27)

### Feat

- add event to toggle event logging
- report generating library
- plotting module for offline data

### Fix

- .gitignore report/
- testing-induced bug fixes

## 0.51.0 (2025-01-27)

### Feat

- add filter stage to pipeline
- add conditional filter

## 0.50.0 (2025-01-25)

### Feat

- add comparator xform

### Fix

- register comparator xform

## 0.49.0 (2025-01-25)

### Feat

- sanitize actions in deployment async env before writing to opc

### Fix

- assert that action tags have operating range specified in config

## 0.48.0 (2025-01-25)

### Feat

- add option to warmup pipeline upon interaction init

## 0.47.9 (2025-01-25)

### Fix

- force all checked configs to be interpolated and non missing
- WIP config fix

## 0.47.8 (2025-01-24)

### Fix

- make delta xform temporal state a dataclass
- maintain temporal state for all sub-xforms in split list

## 0.47.7 (2025-01-24)

### Refactor

- remove pipeline hooks from sim_interaction

## 0.47.6 (2025-01-24)

### Fix

- remove maxlen from event bus queue
- start step cycle immediately when event bus is enabled

## 0.47.5 (2025-01-24)

### Refactor

- make sure calls to update_buffer() and load_buffer() pass PipelineReturn objects
- ensure each agent's load_buffer() method takes a PipelineReturn as an argument instead of a list of transitions
- agent's update_buffer() method now takes a PipelineReturn argument instead of a list of transitions

## 0.47.4 (2025-01-24)

### Fix

- wrap heartbeat opc conn in context manager

## 0.47.3 (2025-01-24)

### Fix

- force dtype conversion in db backup script
- prefer float64 over float_
- remove incompatible ndarray type annotation

## 0.47.2 (2025-01-24)

### Fix

- updated grafana obs query

## 0.47.1 (2025-01-24)

### Fix

- nuke unnecessary runtime validation logic
- add max priority by default as per the PER paper
- define feed and load functions as returning impacted idxs

### Refactor

- migrate ensemble buffer to discrete-dists
- migrate prioritized buffer to discrete-dists
- pull prioritized buffer into own module file
- pull uniform buffer into own module file
- migrate uniform buffer to discrete-dists
- rename sample_batch -> full_batch
- simplify buffer sampling logic
- pull base replay logic out of uniform replay

## 0.47.0 (2025-01-24)

### Feat

- implement preprocessor stage to unify normalization and other early transforms

### Fix

- interaction calls pipeline stages for action ordering and denorm
- now look in ac and preprocess for action denorm

### Refactor

- all arbitrary invertible xforms in preprocess stage
- move reward construction to own dataframe in pipeframe
- move normalization/denorm responsibility to preprocessor
- move action ordering responsibility to AC

## 0.46.3 (2025-01-23)

### Fix

- fix ensemble reduction config
- Fix ensemble reduction config schemas

## 0.46.2 (2025-01-23)

### Fix

- prevent dep_async_env from hanging forever

### Refactor

- replace opc_tsdb_sim with deployment

## 0.46.1 (2025-01-23)

### Fix

- update dv pilot configs

## 0.46.0 (2025-01-23)

### Feat

- minimal (minimal) communication between fastapi and vite
- updated Dockerfile to build and serve client
- using vite and react for client
- base fastapi app with test

### Fix

- allow client/dist to not exist

## 0.45.1 (2025-01-23)

### Refactor

- offline training hooks won't accept any arguments - the objects will already store the data they need
- converted offline_training/utils into a class so that we can ultimately hook evaluators to it

## 0.45.0 (2025-01-23)

### Feat

- implement delta xform constructor

## 0.44.0 (2025-01-23)

### Feat

- add masked autoencoder imputation strategy

### Fix

- label missing temporal state types and fix nullable bug

### Refactor

- setup imputers in base/factory pattern
- create imputer stage group and redirect to per-tag-imputers
- promote imputer to operate across all tags

## 0.43.0 (2025-01-23)

### Feat

- implement add binary xform

### Fix

- update epcor reward to use OR constraint
- shift scrubber reward from [0,1] to [-1,0]

### Refactor

- treat scrubber reward cost minimization generally
- scrubber reward deals with cost in single tag config with add binary xform
- scrubber reward pumpspeed penalties based on generalized constraint violation
- epcor scrubber efficiency reward generalized
- begin to generaliz scrubber reward computation

## 0.42.0 (2025-01-23)

### Feat

- implement min/max binary xform

### Refactor

- rename prod xform to binary

## 0.41.0 (2025-01-23)

### Feat

- wire in ping_setpoint event to dep interaction
- add `ping_setpoint` event

### Fix

- make setpoint pinging optional
- only increment agent step on get_obs

## 0.40.0 (2025-01-23)

### Feat

- add heartbeat thread

### Fix

- increase telegraf write freq

## 0.39.0 (2025-01-23)

### Feat

- dump agent weights to disk every hour

### Fix

- load historical chunk in event bus loop

## 0.38.0 (2025-01-22)

### Feat

- A cli tool to emit zmq events for the event_bus (#408)

## 0.37.0 (2025-01-22)

### Feat

- ema filter queues data to initialize statistics

## 0.36.0 (2025-01-22)

### Feat

- added queue, more reliable cleanup
- added event bus controlled step_event functionality

## 0.35.0 (2025-01-22)

### Feat

- raw data analysis eval
- sketch of eval

### Fix

- rm decorator to remove cicular imports

## 0.34.0 (2025-01-22)

### Feat

- adding agent step

### Fix

- added agent_step to SQL

## 0.33.0 (2025-01-21)

### BREAKING CHANGE

- this removes the concept of `bounds` from the tag
config, instead renaming these to `operating_range` for consistency.

### Feat

- infer normalizer bounds from tag zones
- add yellow, red, and black zones for tags

### Fix

- expect actions to always be normalized -> denormalized
- use identity oddity filter by default
- ensure action tag ordering matches action np.ndarray ordering
- allow xform cfgs to be mutated in place
- prefer Field over field for config schemas
- use discriminated union matching for interaction types

### Refactor

- use Maybe and find utilities to simplify action denorm
- add find utilities for simplifying iterables
- add Maybe monad utility to clean up Nones
- move cfg construction responsibility out of sc to pipeline

## 0.32.3 (2025-01-21)

### Fix

- `opc_tsdb_sim_async_env` now uses `dep_interaction` instead of `sim_interaction`. (#403)

## 0.32.2 (2025-01-21)

### Fix

- adding no nan filter to pipeline tests
- moving nan filtering to transition filter

## 0.32.1 (2025-01-21)

### Fix

- calculate moving avg in one pass
- return nan when mean and std are none
- style issue
- remove small floating point in variance

## 0.32.0 (2025-01-20)

### Feat

- slowly load historical data into buffer

### Fix

- allow norm xform to handle all nan dfs

## 0.31.0 (2025-01-20)

### Feat

- buffered writes
- metrics writer that outputs to csv

### Fix

- added metrics fields to mountain_car_continuous.yaml
- added name field to metric configs
- pulled metrics config out of gym.yaml

## 0.30.1 (2025-01-20)

### Fix

- fix slicing issue, do not impute values when there is no gap

## 0.30.0 (2025-01-20)

### Feat

- adding reward metric in sim interaction

## 0.29.0 (2025-01-20)

### Feat

- rename message_bus to event bus and enable AppState event_bus for when agent emits messages
- added event-bus driven step to interaction and tested on opc_mountain_car_continuous
- started implementing zmq publisher and subscriber
- add pyzmq dependency

### Fix

- Changed scheduler process to thread to use zmq inproc transport
- agent now emit proper event types

## 0.28.4 (2025-01-17)

### Fix

- ensure metrics writer gets closed

### Refactor

- allow directly loading configs e.g. for tests

## 0.28.3 (2025-01-17)

### Fix

- upgrade cenovus configs to latest schema

### Refactor

- allow directly loading configs e.g. for tests

## 0.28.2 (2025-01-16)

### Fix

- swap terminate and truncate in model_env tests
- dispatch on model types correctly

## 0.28.1 (2025-01-16)

### Fix

- grab action from post step instead of prior

## 0.28.0 (2025-01-16)

### Feat

- write features from states encountered online to the metrics table
- make low watermark for metric writer tunable

## 0.27.2 (2025-01-16)

### Fix

- do not reset temporal state on every step
- extend valid threshold for timedelta between pipeframes
- narrow types of first and last timestamp in pipeframe
- remove obs_period from tolerance for stale state in deployment interaction

## 0.27.1 (2025-01-16)

### Fix

- added hypertable and compression config to base_telegraf.conf

## 0.27.0 (2025-01-16)

### Feat

- perform sanity checking on interaction state

## 0.26.5 (2025-01-16)

### Fix

- ensure that large e2e tests have tsdb instance running
- added a dummy writer for unit testing purposes
- use config defined ip instead of hard coded localhost
- wrap try_connect in context manager, always close conn between reads

## 0.26.4 (2025-01-15)

### Fix

- Grafana mounts configs on compose (#383)

## 0.26.3 (2025-01-15)

### Fix

- clean up timing logic in deployment_interaction.py, syncing obs and actions to a shared step_timestamp

## 0.26.2 (2025-01-15)

### Fix

- added metrics, improved grafana aggregation queries

## 0.26.1 (2025-01-14)

### Fix

- allow specifying a node identifier for tags

## 0.26.0 (2025-01-14)

### Feat

- log q-values, actor loss, and sampler loss

## 0.25.0 (2025-01-14)

### Feat

- add config to test deployment env and sim with mountain car

### Fix

- update last_obs_timestamp based on next_obs_timestamp in deployment env

## 0.24.1 (2025-01-14)

### Refactor

- make constructors responsible for column ordering
- plumb ColumnDescription to agent
- pull shared construction logic out of AC and SC
- put all RL constructors in same module

## 0.24.0 (2025-01-14)

### Feat

- build action constructor stage

### Fix

- grab denormalize bounds from normalizer config
- reset sc transform states after dummy data

### Refactor

- remove tag_configs input to interactions
- replace is_action with action_constructor
- get state/action dims and col names from pipeline

## 0.23.0 (2025-01-14)

### Feat

- add metrics logging defaults to all environments
- wire app_state through agents for top-level stateful objects
- add metrics writer as a buffered writer

### Fix

- use docker hostname for timescale ip resolution
- rename dv pilot db config from sensor_table_* to table_*
- table_schema should not be an empty string

### Refactor

- pull buffered writing into generic utility

## 0.22.0 (2025-01-14)

### Feat

- allow specifying configs with leading base in name
- allow specifying a config path with trailing .yaml

## 0.21.0 (2025-01-14)

### Feat

- add interaction factory

## 0.20.1 (2025-01-14)

### Fix

- Add Grafana to Docker Compose (#365)

## 0.20.0 (2025-01-14)

### Feat

- add deployment interaction and env

## 0.19.0 (2025-01-13)

### Feat

- update data reader to include data on the bucket end time and exclude data on the bucket start time
- update opc_mountain_car_continuous config to use reward constructor

### Fix

- optionally provide start time to load_offline_transitions, change offline training test to use a start time obs_period before the first step
- handle gym reward as a special case with tag name gym_reward

## 0.18.0 (2025-01-13)

### Feat

- enable docker builds on release

## 0.17.3 (2025-01-10)

### Fix

- fix circular import issues with product and split transforms

## 0.17.2 (2025-01-10)

### Fix

- add a default 50ms buffer to OPC TSDB sim async env obs read

## 0.17.1 (2025-01-09)

### Fix

- enable serialization of timedelta objs with json dumper
- update tests to use action_period and obs_period as timedelta objects
- use timedelta obs_period in offline utils
- interpolate obs/action_period in data pipeline config schemas; compute steps_per_decision based on obs/action_period in countdown and transition creators
- unify notion of bucket_width, env_step_time, clock_inc as obs_period; refactor env configs to use multiple inheritance to share common fields
- remove action_period and obs_period from top level configs

## 0.17.0 (2025-01-09)

### Feat

- add power transform

## 0.16.0 (2025-01-09)

### Feat

- add generic utility to sort lists at multiple levels

### Fix

- ensure action values are sorted alphabetically
- annotate test utility types

### Refactor

- make tag sort order more explicit

## 0.15.0 (2025-01-09)

### Feat

- bump

## 0.14.1 (2025-01-09)

### Fix

- resolve circular import of TransformConfig type
- use the union type of all xform configs for other xform
- use type annotations to denote discriminators within lists

## 0.14.0 (2025-01-09)

### Feat

- Mockup OPC configuration wizard  (#348)

## 0.13.0 (2025-01-08)

### Feat

- **docker**: first pass at dockerization of corerl

## 0.12.0 (2025-01-08)

### Feat

- add placeholder offline training config for Cenovus
- add utility script to generate Cenovus tags.yaml
- add ability to use paths for default configs
- add ability to download cenovus data from s3

### Fix

- **cenovus**: add script guards to cenovus scripts
- make calling offline_training.py with 0 offline steps an error

## 0.11.0 (2025-01-08)

### BREAKING CHANGE

- this commit removes the Quality column from our
telegraf configuration as well as from our data reader and writer.

### Refactor

- remove "Quality" column from telegraf, data reader & writer

## 0.10.1 (2025-01-07)

### Fix

- ensure behavior policy is updated for InAC

### Refactor

- remove several unused properties on three_tanks
- remove several unused properties on three_tanks_v2
- remove several unused status variables in LSO
- remove reinforce algorithm
- nuke hooks subsystem

## 0.10.0 (2025-01-07)

### Feat

- Utility to convert Drayton Valley DB Backups into a format usable by our OPC Client (#340)

## 0.9.2 (2025-01-07)

### Fix

- add missing type annotations to activations
- add missing type annotations to e2e directory
- make dispatch target functions anonymous
- remove pandas for temporary data structures
- remove unreachable code branch in to_np utility
- remove unnecessary isinstance in scrubber reward

### Refactor

- nuke plotting utilities

## 0.9.1 (2025-01-07)

### Fix

- fully widen internal recursion type and force type-guarding
- tighten up data utility type annotations
- prefer type-narrowing by elimination
- add type annotations for interior HOFs
- add missing type annotations to deprecated scrubber reward
- add missing type annotations for optimizers
- add missing type annotations to tests
- add missing type annotations to sql_logging
- add missing type annotations for environments
- migrate OpcConnection to schematized configs
- add missing type annotations for networks
- add missing type annotations for distributions
- add missing type annotations to buffer
- remove update_priorities function from non-prioritized buffer
- add missing type annotations in agents
- annotate hooks/when types
- annotate missing policy argument types
- add missing type annotations to tests
- add type annotations to transition filter
- add type guard for pf.transitions is None in transition filter
- use literal string names for transition filters
- widen dictionary type
- use covariant Sequence[T] over list[T]

### Refactor

- unthread hooks from four_rooms env

## 0.9.0 (2025-01-06)

### Feat

- working opc_tsdb continuous mountain car

### Fix

- denormalize the ndarray action during emit_action
- make_configs should respect env action & obs bounds

## 0.8.0 (2025-01-06)

### Feat

- define meta tags, support multi dim actions

### Fix

- made make_configs work with MainConfig, added make config test
- **test**: updated to address epcor tsdb stub test issues

## 0.7.0 (2024-12-23)

### Feat

- epcor tsdb scrubber is working with main

### Fix

- **test**: addressed offline test use of docker container initialization
- **test**: improve boto credential test/skip logic
- **test**: do not share module/class fixture of tsdb docker container
- **env**: support temporal state, batch reads with previous ts
- updated saturation config to match gym_name config change
- enable running epcor_tsdb_scrubber with main

## 0.6.5 (2024-12-21)

### Refactor

- rename NewTransition -> Transition
- allow feeding a list of transitions to buffer
- remove hooks from greedy_ac
- remove unused guardrail bounds
- unthread several unused functions now that gac doesn't own delta actions

## 0.6.4 (2024-12-20)

### Fix

- fix redundant variable declaration (again)

## 0.6.3 (2024-12-20)

### Fix

- remove extra fields from transition_creator configs

## 0.6.2 (2024-12-19)

### Refactor

- remove third-party gym environments
- cleanup final uses of hydra and omegaconf
- migrate entirely from gym to gymnasium
- delete more unused code

## 0.6.1 (2024-12-19)

### Fix

- **test**: enable data loader tests, address warnings (#327)

## 0.6.0 (2024-12-19)

### Feat

- use config schema for deployment async env
- add explicit config schemas for environments
- add config option to allow extra fields

### Fix

- handle both dataclass instances and types in dict conversion
- dont prepend 'public' to table schema

### Refactor

- move tagdbconfig defaults into parent class
- move opc simulation scripts to new config framework
- move opc_client to new config framework

## 0.5.2 (2024-12-19)

### Refactor

- pull empty dataframe handling out of stages and into pipeline
- simplify bound_checker implementation
- use shared tag_ts accessor utility
- move all pipeline type def'ns to datatypes

## 0.5.1 (2024-12-19)

### Fix

- ensure fundamental missingness stage is never skipped
- add tqdm to main.py for logging

### Refactor

- use shared temporal_state accessor util for copy and linear imputers
- remove unused boilerplate code from imputers
- remove unused tag_cfg from imputer constructors
- rename agent_transition_creator -> transition_creator

## 0.5.0 (2024-12-19)

### Feat

- **docs**: added autogenerated documentation (#322)

## 0.4.0 (2024-12-19)

### Feat

- register saturation as a gym environment

### Fix

- rewrite saturation configs to new config system

## 0.3.3 (2024-12-18)

### Fix

- **datareader**: agg read should not raise exception if df is empty (#320)

## 0.3.2 (2024-12-18)

### Fix

- **datareader**: addressed aggregated_read NaN, use orm query builder (#319)

## 0.3.1 (2024-12-17)

### Fix

- support ordering of tz naive time buckets, default offline pipeline kwarg (#318)

## 0.3.0 (2024-12-17)

### Feat

- add utility to walk a dataclass and convert to dict
- add utility to get deeply nested values from dict
- add utility to set a value at a deeply nested dict key
- add utility to perform deep merges of dictionaries

### Fix

- everyone gets an __init__.py
- add __init__.py to environments
- add __init__.py to configs
- add agent type name to config
- cleanup extraneous objects in configs
- correctly label agent config as a discriminated union
- ensure config is importable with no modifications
- re-enable a subset of the e2e tests
- ensure consistent datatypes of cols in tests
- remove reactor test

### Refactor

- migrate deployment_manager and event_bus to new config system
- migrate all hydra configs over to new config system
- have bound checker only take bounds as inputs instead of full config
- replace hydra with interal config lib

## 0.2.0 (2024-12-16)

### Feat

- merging `staging2` changes onto `master` (#315)

## 0.1.0 (2024-12-16)

### Feat

- make anytime tc aware of termination and truncation
- expose a simulated gymnasium.env that uses OPC communication protocol (#302)
- try to sync clock with future data if it exists
- have SC track decision points on PF
- make no countdown default for SC
- add action period countdown to SC
- add utility to access tag-level temporal state
- add interaction wrapper for capturing latest state
- added fake OPC client for emitting temporally consistent offline data (#295)
- added fake OPC client for emitting temporally consistent offline data (#295)
- allow querying pipeline for expected state/action dimensions
- add time chunking utility to time lib
- add hardcoded query to get earliest and latest timestamps in tsdb
- hard-code query for obtaining column stats
- make normalization default for tag sc
- add ability to execute arbitrary queries against tsdb
- minimal compose with e2e opc, telegraf, timescaledb (#280)
- add ability to selectively enable and reorder pipeline stages
- ensure queried dataframes contain all timesteps within interval
- add ability to mandate expected split passthrough behavior
- add split state construction
- add ability to concatenate raw observations as a sc step
- add tag to sc tranform context carry object
- implement normalizer state_constructor
- implement trace state_constructor component
- enable reward evaluator by default for pendulum and saturation
- add large saturation.yaml test with CI (#250)
- add change action option to GAC. Not tested yet
- specify tag config attributes
- add datareader option to aggregate using last value in bucket
- allow specifying multiple groups for hydra group dispatcher
- allow specifying multiple groups for hydra group dispatcher
- add utility for one-off structured configs
- automatically build sensor table on first write
- use ParamSpec to enforce types on dispatch functions
- specify expected keys for normalizer config group
- use discriminated union for dispatching group configs
- migrate experiment configuration out of yaml files
- allow interleaving actor/critic updates
- add new `ThreeTanksv2` environment
- make core-rl pip installable as a package
- add four rooms environment
- adds a docker compose file to run a timescaledb container
- add data reader to read from timescale db
- spawn event bus from deployment_manager
- allow client to subscribe callbacks to an event
- add small utility to unify sync and async callbacks
- use new network construction for all nets
- data writer writes in batches, use iso format for timestamp strings
- implement `write` function in DataWriter, create `sensors` table in DataWriter test
- add fixture to setup and remove timescaledb docker container for tests
- add docker compose and handle setup/teardown in data_writer test file
- add stubbed out data writer
- add simple utility to check bounds on Box spaces
- add FTA
- add lifecycle event publishing from agent code
- add basic pub/sub event bus
- add simple time keeping utility
- use configurable timeout for reconnect
- allow configuring deployment manager from a cfg file
- add dictionary hashing utility
- add nullability default util
- add script for keeping main.py alive in prod

### Fix

- set x-access-token actor explicitly (#312)
- commitizen bump and release with gh app (#311)
- **commitizen**: added on push master version bump and release (#309)
- add last_row to temporal_state to track action changes over pfs
- add assertion that state matches last observed state
- make mock action period match mock data
- prefer explicit value for dp on Step
- rename RAGS->Step in e2e tests
- ensure state columns have consistent ordering
- use torch.allclose instead of torch.equal for floating point tolerance
- tab __eq__ to be in NewTransition class
- warn about high watermark before forcing sync
- add more informative no variability error message
- hardcode pipeline default order to ensure consistency
- add bounds checker to pipeline stages
- specify expected time interval in cfg
- fix type error in opc_server
- manually construct tag_ts for sc for each tag independently
- ensure sc components are registered to hydra group
- broadcast new missing info flag over mask
- fix type annotation for test utility
- add type annotations for activation configs
- annotate missing policy types
- annotate missing softmax policy types
- add required keys to cartpole env config
- allow multiple heads for discrete actions policies
- use empty list instead of null for test_epochs
- add caller code to stateconstructor test
- don't require _inner_call for all imputers
- add caller_code to sc test
- remove unnecessary runtime guard
- add config
- fix type error from poor pandas annotations
- fix several minor pandas type errors in tests
- use str over Hashable
- dont inherit from MutableMapping for main_config
- remove unnecessary asserts
- fix incorrect import path
- add type annotations to composite eval
- do not assume sc config has a warmup period
- label alert transition creator config type
- have common base config for state constructor
- use more specific agent config in factory
- apply more specific type annotations to networkactor
- fully erase list_ type
- fill in known policy default configs
- remove unused argument from interaction constructor
- attach only_dp_transitions flag to BaseInteraction due to overrides in main
- remove circular import guards
- remove unsafe kwargs param
- make avg the default aggregation strategy, plus some formatting
- missing import
- use string literals to indicate aggregation options
- use common base class for network configs
- ensure child and parent classes have matching inits
- use optimizer config types for IBE configuration
- remove propagation of vmap flag through code
- use narrower type interface for main utilities
- call super class init from child init
- don't mark non-empty function as abstract
- delete unusable __init__ methods
- widen factory type for config
- add type annotations to action_gap eval and fix related bugs
- fix integration errors in types with dataloader cfgs
- fix type integration with async db writer implementation
- fix type errors in integration with db-cfg branch
- assert sweep gives back dicts
- fix config type declarations for make_offline_transitions
- remove gymnasium.Env from state_constructor
- forward declare alert_tc
- forward declare normalizer types
- fix knock-on type errors for idxing numpy arrays
- use specific cfg type for transition normalizer
- remove unused test runner
- fix nullability errors in calibration_model
- use specific cfg type for transition normalizer
- propagate legacy types throughout utility functions
- use type overloads to avoid unnecessary type widening
- delete unused test file
- fix linting errors in reintegration with master
- no mutable default args
- no bare except
- prefer isinstance over type(x) == T
- avoid datetime.now as default function argument
- ensure all actors have a policy defined
- specify argument types for reductions factory
- specify argument types for network factory
- fix type bug masked by implicit Any argument types
- specify argument types for actor factory
- add get_log_prob to BaseActor interface
- access cfgs via attributes instead of by key
- set 'offline_data_path' to default to 'offline_data' to avoid type checking errors
- added offline_data_path str type assertion for pyright
- needed to update some tests and some config files to be compatible with the separate old_data_loader and data_loader config groups
- updated test_direct_action_data_loader.py to use new data loader factory
- removed deprecated data_loader factories
- updated get_test_state_qs_and_policy_params() to be consistent with Samuel's policy refactor
- fix type error masked by config implicit Any
- remove dead code
- fix type errors in utils/plotting
- fix type errors in tde eval
- fix type errors in q_estimation eval
- fix type errors in policy_improvement evaluator
- fix nullability type bugs masked by yaml configs
- remove action bounds check for identity normalizer
- fix small type bug masked by config implicit Any
- cast return types to match function contract
- ensure base class and child classes match
- ensure sql_logging is included in packaging
- fix missing arguments bug in policy creation
- fix get_dist_type type annotation
- fix float vs tensor bug in policy creation
- fix lingering type errors in influx_opc_env
- remove misleading_reward missing parameter
- use torch.types._size to match base class signature
- bump pytorch version
- fix DataFrame vs Series errors in data loader
- allow row_to_transition to take a dataframe or a series
- filter nullability issue in deprecated anytime interaction
- ignore hard to fix type errors in reseauenv
- add stubbed methods for reseauenv
- cast numpy array to float scalar
- import torch optimizer from originating module
- ensure tests are importing from medium dir
- remove extra test main files
- clean up agent event emitter when process ends
- ensure background processes are nonblocking
- resubscribe to events if client disconnects
- fix return type for get_sweep_params
- fix types and styles in plotting
- add type annotations to state_constructor and components
- delete nonexistant component
- delete functions that do not currently work
- avoid circular imports in type checking
- remove OfflineAnytimeInteraction
- fix nullability issues in anytime_interaction
- use old deprecated transition creator logic for old deprecated interaction logic
- ensure base class and child class match signatures
- import from torch.optim.optimizer directly
- use one-off handling for softmax policies
- use absolute imports to help type checker
- ensure base class and child class match signatures
- import constraints directly from underlying module
- fix knock-on type errors
- guess at value of undefined variable
- ensure variables are defined before use
- fix linesearchoptimizer type errors
- specify strict=True for zip
- switch Optional type annotation for dict
- fix several poor torch type annotation issues
- ignore type errors coming from bad torch type annotations
- ensure all variables are defined before use
- remove non-implemented network types from factory
- remove outdated actor init function
- ensure base class and child class match signatures
- avoid mutable default params
- specify evaluation function parameter type
- specify strict=True for zip
- ignore type errors for bad torch type annotations
- match base class types to instance
- remove mutable default params
- handle nullable batch_size in trajectory buffer
- roll forward obs type errors
- clarify Transition obs types
- fix call signature for factory functions
- pass policy's model to linesearchopt
- ensure override matches base class
- require initializer to be networkactor
- pass dummy closure for optimizer.step
- fix parameter types for dependency functions
- ensure buffer sampling types are consistent
- ensure model and optimizer are defined in init
- remove type errors in anytimecalibrationmodel
- remove type errors in basecalibrationmodel
- ensure sc_agent is defined
- add type annotations to BaseCalibrationModel
- import `override` from `typing_extensions`
- fix linter issues in sql_logging
- make one_step_model compatible with dataclass version of Transition
- ensure base class and child class match signatures
- address undefined references
- remove `mean` from `ArctanhNormal`
- massage sum type to be a tensor
- widen infered dictionary type for stats
- fix type errors for ensemble losses
- ignore type errors from parameters(independent=True)
- remove incorrect redefinition of output from eval child classes
- correctly use return_idx to return a given index
- handle non-gym environments that implement same interface
- fix type/styles in three_tanks
- ensure base class and child class match signatures
- remove mutable default params
- ensure base class and child class match signatures
- fix types/styles of influx_opc_env
- ensure base class and child class match signatures
- fix overloading variable with multiple types
- fix type/style errors in discrete_control_wrapper
- move third_party code into dedicated dir and ignore errors
- fix errors d4rl wrapper
- fix type and style issues in data_loaders utils
- fix type and style issues in transition_load_funcs
- use function overloads to clarify normalizer types
- fix several type and style issues in direct_action.py
- split incompatible dataloader apis
- reduce line length warnings
- remove several nullability errors in transition_creator
- indicate that composite alert can receive None cfg
- ensure composite alert defines all abstract methods
- no overshadowing base class methods with different signatures
- allow torch utility code to take one of many types of device
- remove dead code path
- avoid sum for tensors due to implicit broadcasting
- use prev `create_base` for `EnsembleFC`
- allow ensemble utility to take a single loss
- remove unused imports
- add missing type annotations
- remove default value bug in get_batch_actions_discrete
- unify function overload types
- ensure all abstract methods are defined on concrete instances
- destructure return tuple from get_log_prob
- allow passing a device to Float tensor
- avoid reassigning properties with different types
- ensure ensemble updates take list of losses
- remove unreachable code paths
- use `output_dim` for final network layer
- ignore incorrect torch type interface
- compute_sampler_loss return type is not tuple
- dont return when return type is None
- remove string interpolation for logging
- ensure linesearch versions of objects are being used
- ensure loop variables are bound in closure
- narrow type annotation for q factory
- fix typing of critic update closure
- unify sampling return type
- migrate to logger for warnings
- ensure con_cfg is a dictionary
- add missing type annotations
- add missing type annotations
- add some redundancy when the event loop disappears
- add missing type annotations
- warn when inverting with a non-invertible normalizer
- annotate missing types in transitioncreator
- inline nullability checks for type inference
- use OldAnytimeInteraction for old trans creator
- return cached values to guarantee not null
- ensure pbar variable is defined along all codepaths
- add missing type annotations
- axis->dim for torch methods
- add missing type annotations
- assert Trajectory has been initialized before caching
- use ndarray instead of array for type annotations
- don't reshape state batches to `(0, ...)`
- use more aggressive closing methods on server
- use asyncio.shield instead of custom cancel prevention
- ensure spec.loader exists before accessing
- nuke unused code
- add missing type annotations
- add nullability type checks
- widen _prepare type to arbitrary iterable
- add missing type annotations
- main does not have any return value
- avoid unnecessary type widening
- add complete type annotation to create_obs_transitions
- fix ambiguous return type for get_offline_obs_transitions
- create_obs_transitions returns list instead of dict
- fixup type errors masked by caching func
- offline_alert_training does not return anything
- use tuple[...] instead of (...) for type annotations
- use covariant container type and allow Path
- dont assume all buffer types exist on all algorithms
- allow dictionary configs
- run subprocess outside of shell
- require Path instead of str for saving/loading agents
- strictly return a Tensor from critic functions
- do not raise NotImplementedError in init
- use TensorLike utility to allow lists of tensors
- disallow constant initialization due to incompatible function contract
- fix typo in exception
- use correct Callable type annotation
- use torch.device annotation for device arguments
- change tuple annotations from (a, b) to tuple[a, b]
- add use_alerts config option for gym
- use new agent_transition_creator config option for gym
- typo refix->prefix
- transition creator assumes compositealert is always built
- add default experiment config values for gym.yaml
- delete softmax policy configs
- remove env.plot as most envs do not have a plot method
- remove support for python 3.10
- add hydra-joblib-launcher to requirements
- add configuration option for whether to plot
- use consistent api for normalizers
- separate normalizers into invertible and noninvertible
- only warn when bounds are too loose
- type error when falling out of factory conditional
- avoid unnecessarily widening normalizer types
- bump torch version to 2.4
- no longer pass raw boolean to hashing func
- ensure proper type annotation for linear backoff
- change type annotation from any to Any
- widen type hint for flatten_dict
- fix implicit tuple conversion bug for kepware
- address bug introduced in 8e57989a
- add missing return statement
- fix undefined field `self.policy_copy`
- fix bug where flag was not returned

### Refactor

- use explicit countdown config for tests
- wrap default sc configs in SCConfig object for extensibility
- simplify SC ts accessor
- use a single SC across all tags
- move tensor equality util to predictable location
- pull dfs_close into shared utility
- add explicit copy-backwards utility function
- simplify copy imputer implementation
- simplify temporal state init in copy imputer
- pull state_constructor out to non-swappable module
- pull pipeline configs into dedicated submodule
- class name
- organize data processing utilities by context
- move db utils into dedicated module
- have Pipeline dispatch to stages individually for each tag
- all pipeline stages take a tag name at invoke
- buffer utils
- remove several needless indirection functions
- migrate policies to structured configs
- make policy configs match discriminated union pattern
- move policy creation methods to be near each other
- migrate interactions to structured configs
- annotate interaction config types
- reorganize tests so test logic is up front
- migrate rndexplore to hydra structured configs
- switch BaseExploration to protocol
- migrate transition_creator to hydra structured configs
- remove yaml configs for optimizers
- split optimizers into LSO and base optimizer files
- migrate trace_alerts eval to hydra structured config
- migrate test_loss eval to hydra structured config
- migrate tde eval to hydra structured config
- migrate state eval to hydra structured config
- migrate reward eval to hydra structured config
- migrate q_estimation eval to hydra structured config
- migrate policy_improvement eval to hydra structured config
- migrate ibe eval to hydra structured config
- migrate envfield eval to hydra structured config
- migrate ensemble eval to hydra structured config
- migrate endo_obs eval to hydra structured config
- migrate curvature eval to hydra structured config
- migrate actions eval to hydra structured config
- migrate train_loss eval to hydra structured config
- migrate action_gap eval to hydra structured config
- migrate critics to hydra structured config
- pull ensemble reductions into own module
- migrate actors to hydra structured config
- updated dl_group init to match new Group() constructor so that dl_group can contain both OldBaseDataLoader and BaseDataLoader
- combined old_dl_group and dl_group
- created separate config groups and factories for data loaders inheriting from BaseDataLoader and OldBaseDataLoader'
- move data_loader configs into structured configs
- migrate greedy_iql to hydra structured config
- migrate inac to hydra structured config
- migrate simple_ac to hydra structured config
- migrate sarsa to hydra structured config
- migrate sac to hydra structured config
- migrate reinforce to hydra structured config
- migrate random agent to hydra structured config
- migrate iql to hydra structured config
- migrate greedy_ac to hydra structured config
- migrate action_schedule to hydra structured config
- migrate base agent to structured hydra configs
- migrate event bus to structured hydra configs
- use sqlalchemy interpolation for data insertion
- use structured config for setting up dbs
- rewrite normalizers using structured configs
- move action normalization configs into structured configs
- move maybe_parse_event into shared utility
- move process management into dedicated util
- move time conversions into dedicated util
- move get_open_port to shared utility
- move flatten_list into tested utility
- caching function enforces a zero-arg builder
- break pickle saving into dedicated util
- break maybe pickle load into designated util
- allow cached builder func to take args and kwargs
- pull action bound checking into utility
- remove globally unused function
- remove a level of indentation with early returns
- move cfg hashing into utility
- use dictionary hashing utility in business logic
- move opc linear backoff to decorator
- use single pass for dict flaten
- clean up cfg prep for logging to sql
- move generic dict utilities into utils

### Perf

- heavily optimize tsdb hypertable creation
- use numba compilation to speed up traces
- use non-blocking writer pattern for offline data ingress
