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
