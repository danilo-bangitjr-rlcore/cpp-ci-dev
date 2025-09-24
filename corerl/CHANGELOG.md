# Changelog

## [0.151.0](https://github.com/rlcoretech/core-rl/compare/corerl-v0.150.0...corerl-v0.151.0) (2025-09-24)


### Features

* ability to generate test split in load_offline_transitions ([8060e5f](https://github.com/rlcoretech/core-rl/commit/8060e5f291c738a41b76acfcc72d175ccda62007))
* add method to get matching columns for metrics ([20190cc](https://github.com/rlcoretech/core-rl/commit/20190cc2a5102deef30f8e2606320064af912877))
* change _read_by_metric method to retrieve metrics with optional prefix matching ([a796ee7](https://github.com/rlcoretech/core-rl/commit/a796ee7a3c03d764b9a2d912dbe98dacd3097b19))
* **corerl:** allow independent config of AE missingtol ([cd2867d](https://github.com/rlcoretech/core-rl/commit/cd2867d03adf87338f07f18dd98fcc23798d9fdd))
* **corerl:** support logical or/and sympy expressions ([ef92cff](https://github.com/rlcoretech/core-rl/commit/ef92cff688d4fd70da15b18ea8342988d4361a94))
* enhance _read_by_metric to support prefix matching for metrics ([c38d3e7](https://github.com/rlcoretech/core-rl/commit/c38d3e77bffcc63cb70277a111761653ae016908))
* enhance read_by_steo method to support dynamic column selection with prefix matching ([d657392](https://github.com/rlcoretech/core-rl/commit/d6573924edec2d9f68d095047966031ce19d1bf0))
* enhance read_by_steo method to support prefix matching for metrics ([1e30186](https://github.com/rlcoretech/core-rl/commit/1e301869fa5a5d814ae60be9bf6ffebd45e97446))
* enhance read_by_time method to support dynamic column selection with prefix matching ([6ddbbb2](https://github.com/rlcoretech/core-rl/commit/6ddbbb2b27dff17c0568a6eeafa19a487944a93e))
* enhance read_by_time method to support prefix matching for metrics ([929737a](https://github.com/rlcoretech/core-rl/commit/929737a69b348c648c63b1fc394ff71121db8633))
* initial config GET api ([6651d7f](https://github.com/rlcoretech/core-rl/commit/6651d7fd376222ec8c05fe911462ed19a08012f9))
* **PROD-724:** Ensure Sympy Goal.thresh Is In Goal.tag Operating Range ([f8963b9](https://github.com/rlcoretech/core-rl/commit/f8963b973cdb7d5704dd53cffcf7f4c326ef1aaf))
* **PROD-725:** Ensure Goals/JointGoals are self-consistent and don't violate red zones ([8098cbf](https://github.com/rlcoretech/core-rl/commit/8098cbffbb047f10bdfe324fd08707a219fc3c18))
* **PROD-727:** yellow zone vs operating range checks ([6f9778e](https://github.com/rlcoretech/core-rl/commit/6f9778eed7cac96ef904a89a8776f30bbbce8241))
* **PROD-728:** additional yellow vs red zone checks ([19233f9](https://github.com/rlcoretech/core-rl/commit/19233f9f9d2726380340919433e0cd9d6a9495ce))
* **PROD-730:** red zone reflex checks ([5d8f1a6](https://github.com/rlcoretech/core-rl/commit/5d8f1a6c81dbd2c35c943590e5ee5bb98096f5ed))
* **PROD-731:** Check computed tags can be within their operating range ([ebbc7d4](https://github.com/rlcoretech/core-rl/commit/ebbc7d41ced5e319654af71dd6eaa02f8b89ea62))
* **PROD-830:** Opc navigation backend (part 1) ([#1097](https://github.com/rlcoretech/core-rl/issues/1097)) ([3b5ffd4](https://github.com/rlcoretech/core-rl/commit/3b5ffd44296068831a8d4d29e1c8b84477cb258a))
* **PROD-839:** add transition statistics generation to report ([9529f72](https://github.com/rlcoretech/core-rl/commit/9529f72d353fbd1fa1d50905690316531d285f83))
* **PROD-839:** make_transition_statistics_table adds transition_filtered via metrics ([6f2b398](https://github.com/rlcoretech/core-rl/commit/6f2b398fd60667194b0b152d857cdbf8eeb1f380))
* **PROD-839:** update TransitionFilter to include app_state and log filtered transitions ([a0d90d7](https://github.com/rlcoretech/core-rl/commit/a0d90d73fb8e2393849c0855962ab351c43547a5))
* **PROD-840:** add calculation and statistics for goal violation periods ([6a92bc2](https://github.com/rlcoretech/core-rl/commit/6a92bc2d4ce72601143bef367f74b7107e514cbe))
* **PROD-840:** add goal violations table generation in report ([aa201ee](https://github.com/rlcoretech/core-rl/commit/aa201ee14d1226f7d90f7570f7208513a2ee7a96))
* **PROD-884:** Conditional filter is now an oddity detector ([c90bc86](https://github.com/rlcoretech/core-rl/commit/c90bc863f5df4270305779e041d8bacff19e0533))
* **PROD-884:** Moved Conditional Filter Pipeline Stage To Oddity Detection Stage ([db54ec5](https://github.com/rlcoretech/core-rl/commit/db54ec54e72424596c0d60ecc912ee90b9c47c8c))
* **PROD-889:** update metrics logging to specify filter name in transition filtering ([df1ccc0](https://github.com/rlcoretech/core-rl/commit/df1ccc061f12b9fce7fbb87d519ad105a2b369f8))
* **PROD-890:** parse sympy expressions based on parenthesis ordering ([bcb0d6f](https://github.com/rlcoretech/core-rl/commit/bcb0d6f3897228ca22daa5a0d4beb5392557c8fd))
* **PROD-890:** Parsing Sympy Expressions In Correct Order Based On Parentheses ([d437865](https://github.com/rlcoretech/core-rl/commit/d43786544af5afaa41116be9b84eff4c2cb017e9))
* update read method to include prefix_match parameter ([06e0fec](https://github.com/rlcoretech/core-rl/commit/06e0fec88448311e6e9b601471223346aa1917db))
* update read method to pass in prefix matchin ([6dfe83d](https://github.com/rlcoretech/core-rl/commit/6dfe83db1056a89fb5822ba72975455d05c33e55))


### Bug Fixes

* **corerl:** add schema to data_reader queries ([db5eeba](https://github.com/rlcoretech/core-rl/commit/db5eebab7736e43474fad73ad8ae51577679f6dc))
* **corerl:** add schema to data_reader queries ([1a7307c](https://github.com/rlcoretech/core-rl/commit/1a7307cb1754f93f1329db3051e380a94cdbf684))
* **corerl:** timeseries data should not use time as primary key ([e3ba87a](https://github.com/rlcoretech/core-rl/commit/e3ba87a35b9dfa8ffabc8225f4ae656f39897b66))
* offline tests ([67caf35](https://github.com/rlcoretech/core-rl/commit/67caf3558ce401ccde7745fe4733d7dc7d66877b))
* **PROD-815:** update offline_analysis ([402310c](https://github.com/rlcoretech/core-rl/commit/402310c618909bc75519033e7857674179eb5ae6))
* **PROD-845:** don't generate sympy permutation inputs if tags aren't SafetyZonedTags ([96bfd91](https://github.com/rlcoretech/core-rl/commit/96bfd91fcc648962ca4dc6a77f834e85e0b6b247))
* **PROD-949:** conditional filter should only filter opc tags ([6977cfa](https://github.com/rlcoretech/core-rl/commit/6977cfa6671a162d829f52a3633be1b0ce21232c))
* **PROD-949:** conditional filter should only filter OPC tags ([71c18ed](https://github.com/rlcoretech/core-rl/commit/71c18ed76e1b1ccf8f105fb30dd3ab5f90b242b5))
* **PROD-950:** move virtual and delta tag metrics logging to action constructor ([07293f4](https://github.com/rlcoretech/core-rl/commit/07293f4b9fcffe72d4257dd75ed084905c75e0ef))
* remove return type annotation from split_dataframe_into_chunks function ([cba5d1b](https://github.com/rlcoretech/core-rl/commit/cba5d1b4bf61de61e8a68522e6d88b23da8017e7))
* rename _ensemble method to _contiguous_time_threshold ([0f7a6cf](https://github.com/rlcoretech/core-rl/commit/0f7a6cffb83abbe9dcbd3b80faedf5a5bc1245c9))
* zone_violation metrics include tag and will be logged int offline mode ([4b0b70f](https://github.com/rlcoretech/core-rl/commit/4b0b70ff1b9fbc2a63f745bf84a4ea9e192d71a8))

## [0.150.0](https://github.com/rlcoretech/core-rl/compare/v0.149.0...v0.150.0) (2025-09-18)


### Features

* **PROD-889:** update metrics logging to specify filter name in transition filtering ([df1ccc0](https://github.com/rlcoretech/core-rl/commit/df1ccc061f12b9fce7fbb87d519ad105a2b369f8))
* **PROD-890:** parse sympy expressions based on parenthesis ordering ([bcb0d6f](https://github.com/rlcoretech/core-rl/commit/bcb0d6f3897228ca22daa5a0d4beb5392557c8fd))
* **PROD-890:** Parsing Sympy Expressions In Correct Order Based On Parentheses ([d437865](https://github.com/rlcoretech/core-rl/commit/d43786544af5afaa41116be9b84eff4c2cb017e9))


### Bug Fixes

* zone_violation metrics include tag and will be logged int offline mode ([4b0b70f](https://github.com/rlcoretech/core-rl/commit/4b0b70ff1b9fbc2a63f745bf84a4ea9e192d71a8))

## [0.149.0](https://github.com/rlcoretech/core-rl/compare/v0.148.0...v0.149.0) (2025-09-17)


### Features

* **corerl:** support logical or/and sympy expressions ([ef92cff](https://github.com/rlcoretech/core-rl/commit/ef92cff688d4fd70da15b18ea8342988d4361a94))

## [0.148.0](https://github.com/rlcoretech/core-rl/compare/v0.147.0...v0.148.0) (2025-09-15)


### Features

* **corerl:** allow independent config of AE missingtol ([cd2867d](https://github.com/rlcoretech/core-rl/commit/cd2867d03adf87338f07f18dd98fcc23798d9fdd))
* **PROD-840:** add calculation and statistics for goal violation periods ([6a92bc2](https://github.com/rlcoretech/core-rl/commit/6a92bc2d4ce72601143bef367f74b7107e514cbe))
* **PROD-840:** add goal violations table generation in report ([aa201ee](https://github.com/rlcoretech/core-rl/commit/aa201ee14d1226f7d90f7570f7208513a2ee7a96))


### Bug Fixes

* **corerl:** add schema to data_reader queries ([db5eeba](https://github.com/rlcoretech/core-rl/commit/db5eebab7736e43474fad73ad8ae51577679f6dc))
* **corerl:** add schema to data_reader queries ([1a7307c](https://github.com/rlcoretech/core-rl/commit/1a7307cb1754f93f1329db3051e380a94cdbf684))
* **corerl:** timeseries data should not use time as primary key ([e3ba87a](https://github.com/rlcoretech/core-rl/commit/e3ba87a35b9dfa8ffabc8225f4ae656f39897b66))
* **PROD-845:** don't generate sympy permutation inputs if tags aren't SafetyZonedTags ([96bfd91](https://github.com/rlcoretech/core-rl/commit/96bfd91fcc648962ca4dc6a77f834e85e0b6b247))
* rename _ensemble method to _contiguous_time_threshold ([0f7a6cf](https://github.com/rlcoretech/core-rl/commit/0f7a6cffb83abbe9dcbd3b80faedf5a5bc1245c9))

## [0.147.0](https://github.com/rlcoretech/core-rl/compare/v0.146.0...v0.147.0) (2025-09-08)


### Features

* ability to generate test split in load_offline_transitions ([8060e5f](https://github.com/rlcoretech/core-rl/commit/8060e5f291c738a41b76acfcc72d175ccda62007))
* initial config GET api ([6651d7f](https://github.com/rlcoretech/core-rl/commit/6651d7fd376222ec8c05fe911462ed19a08012f9))
* **PROD-721:** time threshes at least obs_period in duration ([e0cfcd8](https://github.com/rlcoretech/core-rl/commit/e0cfcd815bdd6ddb8f783ea3cb121af70d9c295f))
* **PROD-724:** Ensure Sympy Goal.thresh Is In Goal.tag Operating Range ([f8963b9](https://github.com/rlcoretech/core-rl/commit/f8963b973cdb7d5704dd53cffcf7f4c326ef1aaf))
* **PROD-725:** Ensure Goals/JointGoals are self-consistent and don't violate red zones ([8098cbf](https://github.com/rlcoretech/core-rl/commit/8098cbffbb047f10bdfe324fd08707a219fc3c18))
* **PROD-727:** yellow zone vs operating range checks ([6f9778e](https://github.com/rlcoretech/core-rl/commit/6f9778eed7cac96ef904a89a8776f30bbbce8241))
* **PROD-728:** additional yellow vs red zone checks ([19233f9](https://github.com/rlcoretech/core-rl/commit/19233f9f9d2726380340919433e0cd9d6a9495ce))
* **PROD-730:** red zone reflex checks ([5d8f1a6](https://github.com/rlcoretech/core-rl/commit/5d8f1a6c81dbd2c35c943590e5ee5bb98096f5ed))
* **PROD-731:** Check computed tags can be within their operating range ([ebbc7d4](https://github.com/rlcoretech/core-rl/commit/ebbc7d41ced5e319654af71dd6eaa02f8b89ea62))
* **PROD-830:** Opc navigation backend (part 1) ([#1097](https://github.com/rlcoretech/core-rl/issues/1097)) ([3b5ffd4](https://github.com/rlcoretech/core-rl/commit/3b5ffd44296068831a8d4d29e1c8b84477cb258a))


### Bug Fixes

* offline tests ([67caf35](https://github.com/rlcoretech/core-rl/commit/67caf3558ce401ccde7745fe4733d7dc7d66877b))
* **PROD-815:** update offline_analysis ([402310c](https://github.com/rlcoretech/core-rl/commit/402310c618909bc75519033e7857674179eb5ae6))
* remove return type annotation from split_dataframe_into_chunks function ([cba5d1b](https://github.com/rlcoretech/core-rl/commit/cba5d1b4bf61de61e8a68522e6d88b23da8017e7))

## [0.146.0](https://github.com/rlcoretech/core-rl/compare/v0.145.0...v0.146.0) (2025-08-28)


### Features

* **corerl:** enable state layer norm by default ([5006f73](https://github.com/rlcoretech/core-rl/commit/5006f737108e703741b132e9bacf483d502d698a))
* **corerl:** enable state layer norm by default ([11435a7](https://github.com/rlcoretech/core-rl/commit/11435a77a251a88d953d0f3bcee02e5d24f17da0))
* **PROD-702:** check that ranges/bounds aren't empty if they are defined ([86dfefa](https://github.com/rlcoretech/core-rl/commit/86dfefab54a79a849e2c2c774d24e3d2c0035ca6))
* **PROD-702:** checks that ensure ranges/bounds aren't empty if they are defined ([f2916d1](https://github.com/rlcoretech/core-rl/commit/f2916d1c6aa12e670bffc3313f72d5c30f24ef0f))
* **PROD-729:** Make sure tag with red zone reflex has corresponding red zone ([7df1a7d](https://github.com/rlcoretech/core-rl/commit/7df1a7d703e62bfb9ba062ead06db40f31c67887))
* **PROD-729:** making sure tag with red zone reflex has corresponding red zone ([07b95fc](https://github.com/rlcoretech/core-rl/commit/07b95fcb1d5dc66643d10eb6e570b4c4511dd3a0))
* **PROD-804:** BoundInfo Comparison Helper Functions ([0683117](https://github.com/rlcoretech/core-rl/commit/06831172f4dd7dbed6b8583f286e7e8589fd15b9))
* **PROD-819:** log deltaized tags to metrics table ([b63675d](https://github.com/rlcoretech/core-rl/commit/b63675dc9f2aaa9265996cd68fe78fe7de5e608d))


### Bug Fixes

* create BoundsInfo objects for cascade tags ([8cfbddc](https://github.com/rlcoretech/core-rl/commit/8cfbddccd4c24e853b982a8a3b01db97ea019b26))
* update some config validation checks to allow equality ([375f863](https://github.com/rlcoretech/core-rl/commit/375f863f2799507484398c2b7ca92b6d4fcf8bd4))

## [0.145.0](https://github.com/rlcoretech/core-rl/compare/v0.144.0...v0.145.0) (2025-08-26)


### Features

* add bsm1 environment ([36ed624](https://github.com/rlcoretech/core-rl/commit/36ed6249c98f00f021f2bd2c9a191a298018bd6d))
* add bsm1 environment ([4b4ec7a](https://github.com/rlcoretech/core-rl/commit/4b4ec7a77828410627abd655703fcf9c2f244135))
* **corerl:** log rolling reset metrics ([a023aa5](https://github.com/rlcoretech/core-rl/commit/a023aa55b67ad7b20b4569aacf3d0f4a55c3b582))
* layer norm feature flag ([f7b325d](https://github.com/rlcoretech/core-rl/commit/f7b325dae1caae7c90e69dfece1f27be7d9a27e3))
* new reset logic, metric logging and some other minor fixes ([fb6a40d](https://github.com/rlcoretech/core-rl/commit/fb6a40d3ae338d462102058e223464d1dbaa0852))
* new reset logic, metric logging and some other minor fixes ([c905239](https://github.com/rlcoretech/core-rl/commit/c90523955107d2b01c5c615ab4baa6a3d17fa3e3))
* **PROD-763:** rolling reset ([8ae7645](https://github.com/rlcoretech/core-rl/commit/8ae7645199e12085725291d5d9f003062e71dbe8))
* **PROD-763:** rolling reset ([ecb7107](https://github.com/rlcoretech/core-rl/commit/ecb7107226aab29ab192acc2da76ef35155cd213))
* **PROD-790:** Nominal Setpoints Specified As Raw Values ([97b92ad](https://github.com/rlcoretech/core-rl/commit/97b92ad63d951fefa5d7694cee0cce2b54f75bff))
* **PROD-790:** normalize raw nominal setpoint ([5206ad3](https://github.com/rlcoretech/core-rl/commit/5206ad3f55675ee724ffc77fdb53821a42fb3832))
* **PROD-803:** Bounds Refactor ([5535b2b](https://github.com/rlcoretech/core-rl/commit/5535b2b8799fb215496565093801e285aec747dd))
* **PROD-803:** Create BoundFunction and BoundTags types ([42335e5](https://github.com/rlcoretech/core-rl/commit/42335e54ffdb5c477778d7543d883cd3877ea096))
* **PROD-803:** Create BoundInfo class that stores info about individual bounds and BoundsInfo class that stores info about upper + lower bound pair ([70c043e](https://github.com/rlcoretech/core-rl/commit/70c043e8627d9241c9fdd45e215edcbe785ba4b6))
* **PROD-805:** log unnormalized virtual sensor values to the metrics table ([de677fc](https://github.com/rlcoretech/core-rl/commit/de677fc5f8c571c2e638bd5a96d56174c7ca4f59))
* **PROD-805:** log unnormalized virtual sensor values to the metrics table ([822f4ce](https://github.com/rlcoretech/core-rl/commit/822f4cef50f7974a143e952fedcce5d602f3da77))
* **PROD-806:** layer norm feature flag ([5bd0aac](https://github.com/rlcoretech/core-rl/commit/5bd0aacff56c420093611bd3124815be11c8abf3))
* **PROD-813:** add evaluation period to offline training script ([bcc11f5](https://github.com/rlcoretech/core-rl/commit/bcc11f5b62688868618a73a50ebe58f7770f5764))
* **PROD-813:** add option to remove evaluation periods from training ([2623c91](https://github.com/rlcoretech/core-rl/commit/2623c916800b708d281b96c0517d19f33a151532))
* **PROD-813:** add update_agent parameter to run_offline_evaluation_phase ([dc341f3](https://github.com/rlcoretech/core-rl/commit/dc341f31f53f9cc31d8cbb5a72940df83fb6adce))
* **PROD-813:** configurable update agent during rollouts ([09ef87e](https://github.com/rlcoretech/core-rl/commit/09ef87e25a6f32d735ad85b8c64c6968a8b28673))
* **PROD-813:** multiple offline rollouts ([c8f44df](https://github.com/rlcoretech/core-rl/commit/c8f44dfe203691fef8cefc01642236d74c6f1cb6))
* **PROD-813:** offline eval logs to metrics ([68fcd5a](https://github.com/rlcoretech/core-rl/commit/68fcd5ad1d2d3e4f5d09c8e57f59aa0460708e97))


### Bug Fixes

* add assertion to check pipeline output in offline training ([fde89d1](https://github.com/rlcoretech/core-rl/commit/fde89d10ee32b31a6aac49aa145eaf6772555878))
* add logging for offline agent training steps ([b449cfb](https://github.com/rlcoretech/core-rl/commit/b449cfb8cb9d3de1ea6a68dddedabb5a8cd1821a))
* address comments and a divide by zero thing ([813ccb5](https://github.com/rlcoretech/core-rl/commit/813ccb51647d8fd0ec97fc7852e004886db16731))
* address comments and shape mismatch problem ([69a3cbf](https://github.com/rlcoretech/core-rl/commit/69a3cbf0f1dc05d9b1453f617ea3a59eb1b3670d))
* address some comments, and rebase of the critic utils ([23e9776](https://github.com/rlcoretech/core-rl/commit/23e9776e3640582ef6c0ebccf37dfb083f29f588))
* bad indentation ([d1530dc](https://github.com/rlcoretech/core-rl/commit/d1530dc7d4846b9b9ea42a54ebf092b5c829792f))
* **corerl/data_reader:** use sanitized name in final typecasting step ([748a92e](https://github.com/rlcoretech/core-rl/commit/748a92e11e68dafe0f94f8e8adcd97fa94095ad1))
* **corerl:** linting -- copilot's comment ([ce69890](https://github.com/rlcoretech/core-rl/commit/ce698906871873a164eeb52393b1f3cab7fad4ce))
* **corerl:** unify return of metrics read calls ([c8258ce](https://github.com/rlcoretech/core-rl/commit/c8258ce5817d2c3c872497705e5b5932231eb769))
* **corerl:** uses sanitized column names ([2c44b75](https://github.com/rlcoretech/core-rl/commit/2c44b758de0a18e4d6797b0b0b6bcd6c84b82363))
* ensure normalizer config's from_data is properly set after norm bounds are updated ([bc88f6d](https://github.com/rlcoretech/core-rl/commit/bc88f6d5b62d50e02287c4361d5bdaa9c1fd7535))
* ensure there's data to write to metrics table ([e7279c6](https://github.com/rlcoretech/core-rl/commit/e7279c6d890684f2ac7f1c6da10cee71f5715471))
* fix logging ([c905239](https://github.com/rlcoretech/core-rl/commit/c90523955107d2b01c5c615ab4baa6a3d17fa3e3))
* fix priority satisfaction and optimization performance metrics ([c694f11](https://github.com/rlcoretech/core-rl/commit/c694f11bbcedebfb3f6fa8a2e8b1934517a3b00f))
* improve logging format for agent replay buffer sizes ([a983d00](https://github.com/rlcoretech/core-rl/commit/a983d00ff4e3305d5e8b50aed9b1f5fcbc6f7724))
* increment agent step ([e058b53](https://github.com/rlcoretech/core-rl/commit/e058b533e7713156bab694b4c1678a3b2129d37e))
* **lib_agent:** simplify qrc get_active_values api ([eff9486](https://github.com/rlcoretech/core-rl/commit/eff948669e0de4231f90cbff2a9eb11faf621ba9))
* move logging to before possible continue ([c8aee4a](https://github.com/rlcoretech/core-rl/commit/c8aee4a427adb7a60c34cbaa04814379f14ce8ea))
* priority metrics name ([afbb06a](https://github.com/rlcoretech/core-rl/commit/afbb06a3aa8fc4824eea3d2a56c5d1505fee7418))
* **PROD-789:** CoreRL/CoreIO sanitize wide format column names ([07f6f9c](https://github.com/rlcoretech/core-rl/commit/07f6f9cd7a87fde3e806b62d93ea7e0c701c698a))
* remove broken offline eval ([a0365e2](https://github.com/rlcoretech/core-rl/commit/a0365e26ba1d27d2341cedacb867b2df523adfeb))
* remove extra space ([a8b0298](https://github.com/rlcoretech/core-rl/commit/a8b029834d173f8abbb3eef3baac56ab391a84f0))
* remove logging of rundundant priority met metrics ([eb6d3b8](https://github.com/rlcoretech/core-rl/commit/eb6d3b8b90536c9d82773ee4df21ba0627baa895))
* remove unnecessary cast to tuple ([1c675ac](https://github.com/rlcoretech/core-rl/commit/1c675ac4a6b26cdf1a7b2ff65d560eddb6be577e))
* remove unnecessary space in norm_next_a_df clipping operation ([31dec1f](https://github.com/rlcoretech/core-rl/commit/31dec1fe3a71438b5f1c857d6edd3633acb34e95))
* rmove returns so logging down_to possible ([8ee1b75](https://github.com/rlcoretech/core-rl/commit/8ee1b75ae54ed198aa9d533c6be248f42fa1adbe))
* set logging level for offline training ([c184bb1](https://github.com/rlcoretech/core-rl/commit/c184bb189b74f2f1859c20125595a263c9cf3e83))
* styles ([c905239](https://github.com/rlcoretech/core-rl/commit/c90523955107d2b01c5c615ab4baa6a3d17fa3e3))
* styles ([c905239](https://github.com/rlcoretech/core-rl/commit/c90523955107d2b01c5c615ab4baa6a3d17fa3e3))
* styles ([c905239](https://github.com/rlcoretech/core-rl/commit/c90523955107d2b01c5c615ab4baa6a3d17fa3e3))
* styles ([23e9776](https://github.com/rlcoretech/core-rl/commit/23e9776e3640582ef6c0ebccf37dfb083f29f588))
* styles ([23e9776](https://github.com/rlcoretech/core-rl/commit/23e9776e3640582ef6c0ebccf37dfb083f29f588))


### Performance Improvements

* **corerl:** add a single row pipeline benchmark ([aa4b314](https://github.com/rlcoretech/core-rl/commit/aa4b314610cb011af7243bb33eda79ad949b2475))
* **corerl:** measure performance of actor/critic components ([ac38082](https://github.com/rlcoretech/core-rl/commit/ac38082693ff4c1c26d66b36dee633ff04379d8a))
* **corerl:** measure performance of critic queries ([5ec8bba](https://github.com/rlcoretech/core-rl/commit/5ec8bba1caf761ebc3fc84dac6be3fac4e9db89e))
* **corerl:** measure performance of greedyac updates ([769482f](https://github.com/rlcoretech/core-rl/commit/769482f325314ff332dbef1395857882001cb27c))
* **corerl:** remove missing_info from pipeline ([9bb2d9d](https://github.com/rlcoretech/core-rl/commit/9bb2d9d8568c32d395ba358b8a51575ed4f7daa1))
* **corerl:** remove missing_info from pipeline ([96dd3ee](https://github.com/rlcoretech/core-rl/commit/96dd3ee2a014a22f556e979abd70317bd4906c1d))

## [0.144.0](https://github.com/rlcoretech/core-rl/compare/v0.143.0...v0.144.0) (2025-08-13)


### Features

* **PROD-720:** obs_period +  action_period alignment ([6d0d305](https://github.com/rlcoretech/core-rl/commit/6d0d30559f330bd8f5e66d3180c85fa9c202cdde))
* **PROD-726:** check expected_range within operating_range ([e1da182](https://github.com/rlcoretech/core-rl/commit/e1da182cb944be4946df50a8f8b84057db70c7b0))
* **PROD-783:** define operating/expected ranges for cascade tags ([bbf5b3a](https://github.com/rlcoretech/core-rl/commit/bbf5b3a03c76c016d4f6482c1f50f6da22810287))
* **PROD-783:** define operating/expected ranges for cascade tags ([eb33795](https://github.com/rlcoretech/core-rl/commit/eb337953a901880f2fc8e69b75d2638e4c3d8f36))
* **PROD-784:** CoreIO data ingress tests and code coverage ([c4f42c4](https://github.com/rlcoretech/core-rl/commit/c4f42c4a1f2369392ca7a95b8a0847ef55f66460))


### Bug Fixes

* **Coreio:** Coreio's event bus uses callbacks ([9964c06](https://github.com/rlcoretech/core-rl/commit/9964c063bac20a3e1e510763f1bfd3865a62bb8b))
* **corerl:** using the new IO events ([9b3d712](https://github.com/rlcoretech/core-rl/commit/9b3d7124edfc17cbe8a70a5b9ac2ca1b350d57b8))
* correctly name columns throughout test ([5fd3a2c](https://github.com/rlcoretech/core-rl/commit/5fd3a2c1ecb38ceb338f8a6771e76196dcfe34f1))
* **PROD-781:** only use red zone bounds for normalization when they are floats ([eda68be](https://github.com/rlcoretech/core-rl/commit/eda68bef564948c02e0d42bab916848c5d8110c8))
* **PROD-781:** only use red zone bounds for normalization when they are floats ([d22b776](https://github.com/rlcoretech/core-rl/commit/d22b7760be1f6fe293e9a03993890c76f8473c23))
* **PROD-782:** make sure columns that aren't preprocessed don't have column names changed ([8d55476](https://github.com/rlcoretech/core-rl/commit/8d554762cd51e8b8aa82ff1b88fe345a630133e7))
* **PROD-782:** Make Sure Tags That Aren't Preprocessed Don't Have Column Name Changed ([03458c7](https://github.com/rlcoretech/core-rl/commit/03458c79571e5236a73bb33d6f3b1c82a0edfa96))
* updating coverage comment text and uv lock ([d9172ec](https://github.com/rlcoretech/core-rl/commit/d9172ece643441327face048631cf5be40a9d5c4))

## [0.143.0](https://github.com/rlcoretech/core-rl/compare/v0.142.0...v0.143.0) (2025-07-30)


### Features

* **AGNT-82:** enable nominal setpoint bias by default ([7f36e60](https://github.com/rlcoretech/core-rl/commit/7f36e60d3d5abfadb42dcb67b46ba068f4c9aca9))
* **AGNT-82:** enable nominal setpoint bias by default ([0c5cfed](https://github.com/rlcoretech/core-rl/commit/0c5cfed02d7bdf475bf65a376242c406071d5856))
* **CoreIO:** Create and Validate tsdb table for data ingress ([7bd91fd](https://github.com/rlcoretech/core-rl/commit/7bd91fd21239f8d41cc5e4ad5226ea732aa2d3e4))
* **coreio:** new config stubs ([3704bd5](https://github.com/rlcoretech/core-rl/commit/3704bd5d2f20faff8aa28e0730dc7dd233036964))
* **corerl:** actor critic eval fails gracefully ([4dece68](https://github.com/rlcoretech/core-rl/commit/4dece68b2a9c15c4740a1f98704dcb4c1566f857))
* **corerl:** add noisy network feature flag to prod agent ([ca461cd](https://github.com/rlcoretech/core-rl/commit/ca461cd84faf7e25203ff758d0a16d9ad59b269d))
* **corerl:** add red zone reaction configuration ([48b33fd](https://github.com/rlcoretech/core-rl/commit/48b33fdddd92cad26cc81006468be0f26027085a))
* **corerl:** add redzone reflex handling ([f8d43d5](https://github.com/rlcoretech/core-rl/commit/f8d43d5f0d10e6475e451e65a0759712ec718977))
* **corerl:** enable higher critic stepsizes by default ([d848062](https://github.com/rlcoretech/core-rl/commit/d8480621388b6b361b3250e64eedc635c64cb396))
* **corerl:** enable higher critic stepsizes by default ([7c70c42](https://github.com/rlcoretech/core-rl/commit/7c70c42cfd01fcc2fe7ba2097aa2138f070f3a31))
* **corerl:** enable tiny ensemble by default ([38b0427](https://github.com/rlcoretech/core-rl/commit/38b04277f379b040a7a6ad3a8a07fe8afe73750f))
* critic reduction for actor update ([f381e86](https://github.com/rlcoretech/core-rl/commit/f381e86c96928c8ca1d0f293f50c859994015648))
* log predication in mc immediately ([c38167f](https://github.com/rlcoretech/core-rl/commit/c38167fe1763f205046c9477640560d50ab4aafd))
* prod agent use lib agent buffer ([3b2eb05](https://github.com/rlcoretech/core-rl/commit/3b2eb0524fddc73e1eec0cd6a2142a82fea20c76))
* **PROD-502:** weight and grad norm for each layer and each ensemble ([107909e](https://github.com/rlcoretech/core-rl/commit/107909e72cb5e351478dca93ca7704f66c9e8228))
* **PROD-583:** raise validation error on duplicate tag ([627563c](https://github.com/rlcoretech/core-rl/commit/627563c088c59e3b9249ad240e729d2de948353b))
* **PROD-588:** add trace nanmask to AE input ([b57993d](https://github.com/rlcoretech/core-rl/commit/b57993d8099464e39c2761102b8f2ee7363c95ae))
* **PROD-590:** ae simulates missing traces ([c4c190e](https://github.com/rlcoretech/core-rl/commit/c4c190e95ab18389e788344bc631689c359ca28f))
* **PROD-610:** configurable bound checker tolerance ([14e2b27](https://github.com/rlcoretech/core-rl/commit/14e2b2773caa8f92c26683a7d587e7eccfda8ef9))
* **PROD-610:** make bound checker tolerance configurable ([e95a326](https://github.com/rlcoretech/core-rl/commit/e95a326cf1a2a65b29baa9f55b6a5ecfee7a509b))
* **PROD-611:** log trace quality for AE & SC as metrics ([90ad47c](https://github.com/rlcoretech/core-rl/commit/90ad47ce74c3dc1c5cbe0c5a420f9e3215c8f0ae))
* **PROD-611:** track trace quality to handle warmup + nan resilience ([a9d6392](https://github.com/rlcoretech/core-rl/commit/a9d63926ab5b865b00d3f862627686ece65f12e0))
* **PROD-667:** initial version of dashlib ([74e38a3](https://github.com/rlcoretech/core-rl/commit/74e38a387a10cf13d995b9be8f558edbd6ccc801))
* **PROD-677:** log priority metrics ([51a5917](https://github.com/rlcoretech/core-rl/commit/51a59177085d325fd5e479616a214db8cb22f892))
* **PROD-685:** function to get q-values and probabilities for an offline eval state ([3cfecca](https://github.com/rlcoretech/core-rl/commit/3cfecca8b3f85a85848a453f6409c6b11de9d2dc))
* **PROD-685:** function to make actor-critic plots for offline eval states ([94e3ff1](https://github.com/rlcoretech/core-rl/commit/94e3ff1ce54c330211d8f824babc0e8aba80dde4))
* **PROD-685:** offline training actor critic plots ([33b5f30](https://github.com/rlcoretech/core-rl/commit/33b5f309e42e42ef637a151e0054ba41a8b68727))
* **PROD-704:** n samples to get_actions ([d8d6bb5](https://github.com/rlcoretech/core-rl/commit/d8d6bb58adec926802dc81ded4a4e17a09e3f1ca))
* **PROD-712:** abstract hard and soft sync ([68a675d](https://github.com/rlcoretech/core-rl/commit/68a675d2d5b61d077ad7c1bf047475d3517dc249))
* **PROD-712:** buffer len() ([17c9969](https://github.com/rlcoretech/core-rl/commit/17c9969cd70a99a6bc5bf2a7beba98dc61f6093b))
* **PROD-712:** configurable sync ([c7d26e7](https://github.com/rlcoretech/core-rl/commit/c7d26e777bc1727317cf7217483671a959dba339))
* **PROD-713:** sync_group ([00f21be](https://github.com/rlcoretech/core-rl/commit/00f21be27a57e486ec758fc9368f45baffec71c2))
* **PROD-713:** time based sync cond ([645caef](https://github.com/rlcoretech/core-rl/commit/645caefb92244ae11ea0a768f47be6fed17e3e67))
* **PROD-713:** track sync state ([701ea6e](https://github.com/rlcoretech/core-rl/commit/701ea6ed0263d8caf49f57695a8cf8a9a14ae14b))
* **PROD-719:** Ensure start times &lt; end times in offline training ([3e05c7e](https://github.com/rlcoretech/core-rl/commit/3e05c7e927aabb9fffe38f95f6355b969970cf31))
* **PROD-719:** make sure offline start time comes before end time in OfflineConfig ([d31535d](https://github.com/rlcoretech/core-rl/commit/d31535da975ec2d795dcf4110ebcf6ba028d51e9))
* **PROD-722:** ensure AllTheTimeTCConfig is configured for valid on-policy transition creation ([a38a397](https://github.com/rlcoretech/core-rl/commit/a38a3976e206c0276f8ba12b78109fd96ee2f5ad))
* **PROD-722:** Ensure Valid On-Policy Transition Creation ([49bb0ae](https://github.com/rlcoretech/core-rl/commit/49bb0ae1328dc2914a502d7ba52c2eea4594f6cf))
* **PROD-723:** added transition filter compatibility check ([7b2a652](https://github.com/rlcoretech/core-rl/commit/7b2a6520d06cd4decf2e36f3fc7069faca0a8400))
* **PROD-723:** complementary transition filters ([ca2dd00](https://github.com/rlcoretech/core-rl/commit/ca2dd00bfc59c2f6ed4e176125c6204255b8525c))
* **PROD-732:** nominal setpoint must be within operating range ([a7b8b8a](https://github.com/rlcoretech/core-rl/commit/a7b8b8a5158842381b0dbb5d12f167a18db8610d))
* **PROD-732:** nominal setpoint must be within operating range ([05dfbab](https://github.com/rlcoretech/core-rl/commit/05dfbab4a2733b8b38f31c3513fab3278e201f13))
* **PROD-771:** Coredinator fist commit ([8d68d74](https://github.com/rlcoretech/core-rl/commit/8d68d742ce6da5cc098cf336dff48719a2344ab1))
* seperate lr for mu and sigma ([b40a421](https://github.com/rlcoretech/core-rl/commit/b40a421a342ec3b781cb521b57805759a6fccaf5))
* wire up mc eval to interaction ([9d31c07](https://github.com/rlcoretech/core-rl/commit/9d31c07008d461fa903d56f0e60b8bf0e70f1969))


### Bug Fixes

* add dummy_app_state to GC tests ([3a1d19e](https://github.com/rlcoretech/core-rl/commit/3a1d19e4a85b0ff2fb1645a227e429cad36e7c41))
* add lower watermark configuration to EvalDBConfig and MetricsDBConfig ([395029f](https://github.com/rlcoretech/core-rl/commit/395029f3c97d27850eb1586b62adfb39767b9ffd))
* add missing argument to [@post](https://github.com/post)_processor method ([29534fa](https://github.com/rlcoretech/core-rl/commit/29534fadcd39236209ff8b37d89e0f5fcac62486))
* add unit tests for consumer thread util ([1b2fc2b](https://github.com/rlcoretech/core-rl/commit/1b2fc2b28c5ee17531249201f3a66f4709868853))
* address comments ([8c9d2ea](https://github.com/rlcoretech/core-rl/commit/8c9d2ea576169b0eb89e3d4b5eac51994d1b072c))
* address comments ([6ee233a](https://github.com/rlcoretech/core-rl/commit/6ee233abf5100cbcae02dd1b2c849d8362baa946))
* address comments ([6ee233a](https://github.com/rlcoretech/core-rl/commit/6ee233abf5100cbcae02dd1b2c849d8362baa946))
* address comments ([6ee233a](https://github.com/rlcoretech/core-rl/commit/6ee233abf5100cbcae02dd1b2c849d8362baa946))
* address pr comments + rebase ([ca33721](https://github.com/rlcoretech/core-rl/commit/ca33721464362eab0ec574eec702d982eb3db9af))
* address the comments ([e392064](https://github.com/rlcoretech/core-rl/commit/e392064bc9a560b4ca8e41238cab920d566e1705))
* App State no longer uses protocol ([471d90b](https://github.com/rlcoretech/core-rl/commit/471d90b1137f79c0312753e258c87c1602df8640))
* **base_event_bus:** moving publisher socket to base class ([5d5ffeb](https://github.com/rlcoretech/core-rl/commit/5d5ffebd3e08790165d27f5a429845ba3df5600a))
* **BaseEventBus:** move listen forever to base class ([204140c](https://github.com/rlcoretech/core-rl/commit/204140c58a2cceb18c0ed6e7f75750076c5a3b2e))
* check for None operating range ([7cabf5a](https://github.com/rlcoretech/core-rl/commit/7cabf5adf4679b05f1de71abe73a43853493ba3f))
* **CI:** updating uv lock ([57f1fe5](https://github.com/rlcoretech/core-rl/commit/57f1fe5b109c2ff37ded89eb6969ffb8de7ae5fa))
* **coredinator,corerl:** remove unnecesary script ([d46f029](https://github.com/rlcoretech/core-rl/commit/d46f0291bca4f4462dfb596426d4d3150714f98e))
* **coreio:** Refactor communications objects and add minimal SQL integration ([2f45b9a](https://github.com/rlcoretech/core-rl/commit/2f45b9a5335509cd048c401278edcd24bb6fa0b6))
* **corerl, coreio:** update uv lock ([c8c7790](https://github.com/rlcoretech/core-rl/commit/c8c779084a45fb7a39c6d2c072a6188c89a48b34))
* **corerl, lib_utils:** get_sql_engine is now a lib_utils function ([9dbb5a6](https://github.com/rlcoretech/core-rl/commit/9dbb5a617be826e82b23e2dd81468736afb8110b))
* **corerl, lib_utils:** get_sql_engine is now a lib_utils function ([bbf85bb](https://github.com/rlcoretech/core-rl/commit/bbf85bbab38ab95c3064a25657292904a81933bd))
* **corerl, libs:** Move SQL Utils into libs/lib_utils (part 1) ([e49d1b8](https://github.com/rlcoretech/core-rl/commit/e49d1b871417d6c6aa770c8af8cb940746ac637e))
* **corerl,lib_utils:** Generalize definition of BaseEventBus ([5621e56](https://github.com/rlcoretech/core-rl/commit/5621e56ebc75ed083afa36e639cbdaf01c36968f))
* **corerl,lib_utils:** move connect sql engine to lib_utils ([a42d83f](https://github.com/rlcoretech/core-rl/commit/a42d83fb0b7d1c329e64eda0031f8a2b18528079))
* **corerl,lib_utils:** moving sql utils to shared lib part 1. ([b0e2531](https://github.com/rlcoretech/core-rl/commit/b0e2531815288ec4c08c8377be6f7b6377fd8d87))
* CoreRL's event bus now compatible with generic Clock ([693328c](https://github.com/rlcoretech/core-rl/commit/693328cb3cde3a2de1647e208ffe1c63e2af7aaa))
* **corerl:** add basic python dockerfile ([cc21985](https://github.com/rlcoretech/core-rl/commit/cc219857a4e039586fa46cf61b61a623ab94c7c2))
* **corerl:** add more violation information to zone_violation events ([d5f27d4](https://github.com/rlcoretech/core-rl/commit/d5f27d4da27961b2a03e2bc5e251ee032e2a06cd))
* **corerl:** AE only imputes perc missing up to training proportion ([d8b9ddf](https://github.com/rlcoretech/core-rl/commit/d8b9ddf61d60f9d1e980e55d5b130eeb35f5ee79))
* **corerl:** clearer messaging for ai_setpoints that will communicate to coreio ([027daab](https://github.com/rlcoretech/core-rl/commit/027daab9a4837b91e2329365847c84d7ec60c038))
* **corerl:** copy reward in allthetime tc and do not scale reward in interaction ([3cfb82b](https://github.com/rlcoretech/core-rl/commit/3cfb82b0e733cf732a9f740403bf33ef73f74db2))
* **corerl:** dont add cascade deps if already in tag list ([42ab4fa](https://github.com/rlcoretech/core-rl/commit/42ab4fa4a5abd03f5a7d72e8a40d22c53ac537b7))
* **corerl:** dont add cascade deps if already in tag list ([cfbd705](https://github.com/rlcoretech/core-rl/commit/cfbd7056ab972688959894fdac4d4949e84d7b51))
* **corerl:** dont inspect private attributes in tests ([4cf3de7](https://github.com/rlcoretech/core-rl/commit/4cf3de71852822a04132dd699d4008edda09282b))
* **corerl:** fix bug in q eval ([cce1838](https://github.com/rlcoretech/core-rl/commit/cce18388b37ab6c7bd4f827bdcb5419fc87e3b76))
* **corerl:** get a new rng on every representation metric evaluation ([4fc42ea](https://github.com/rlcoretech/core-rl/commit/4fc42ea8a1d6bb100950eb1e65b2df650e07149d))
* **corerl:** never crash on failed checkpoint ([22353d5](https://github.com/rlcoretech/core-rl/commit/22353d5b4e9092aaf2d4feb7f36738840265dd53))
* **corerl:** only construct tagconfig with keyword args ([2417e55](https://github.com/rlcoretech/core-rl/commit/2417e553d6d7094e0cc85ebb9789b75da886cb25))
* **corerl:** only wake up AE if buffer nonempty ([cad9cc5](https://github.com/rlcoretech/core-rl/commit/cad9cc5fc287c69f80b3734778325ae52a489602))
* **corerl:** only wake up AE if buffer nonempty ([88c8b51](https://github.com/rlcoretech/core-rl/commit/88c8b515e433a1d79f4ec9a2dcf5da87a4fc7ade))
* **corerl:** remove side-effect from return normalization feature flag ([2b8b872](https://github.com/rlcoretech/core-rl/commit/2b8b872ae6a6c1fb09bf3d6f4cf4bf6f299f9b40))
* **corerl:** remove unnecessary compose file ([9c161af](https://github.com/rlcoretech/core-rl/commit/9c161afbeea46c34411304b7d8fb2e2db31ef86a))
* **CoreRL:** Rename EventBus events ([edb3d8d](https://github.com/rlcoretech/core-rl/commit/edb3d8d550664d59a13f307827e4bc712864abac))
* **CoreRL:** Rename EventBus events ([f7b356c](https://github.com/rlcoretech/core-rl/commit/f7b356ca7f3d29401ded18b238c45119918d1bb6))
* **corerl:** simplify Clock instantiation with factory function ([30d2c06](https://github.com/rlcoretech/core-rl/commit/30d2c0651339240f43bb5b7b335bc5d469cc52c2))
* **corerl:** updating uv lock ([55c7338](https://github.com/rlcoretech/core-rl/commit/55c7338a3640dabaf9f1949880eaeabe56d45727))
* **e2e:** Docker compose and local corerl/coreio ([c6c50ed](https://github.com/rlcoretech/core-rl/commit/c6c50ed1e1a36f44f882b42c4751f94176a43a4d))
* fix bug in q eval ([a764edc](https://github.com/rlcoretech/core-rl/commit/a764edc777d1d523be0de8d32c5829b5f3218212))
* fixed error messages ([adf1347](https://github.com/rlcoretech/core-rl/commit/adf13478b261df98578d5ac34f9c15eabe4ff60f))
* For CI checks ([7e212ee](https://github.com/rlcoretech/core-rl/commit/7e212ee12dbda7f35e1c741a5698b900970e5f58))
* github action and styles again ([63ad85a](https://github.com/rlcoretech/core-rl/commit/63ad85a401b22a439652f110171ae720cc0fae28))
* **Kerrick's comment:** differentiating socket address strings and socket objects ([e3a54db](https://github.com/rlcoretech/core-rl/commit/e3a54db8323dcf908081eea6bf138f250f3d163f))
* lib agent buffer test and also style fixes ([f6e89cf](https://github.com/rlcoretech/core-rl/commit/f6e89cf50710ed96f752a965849ae7faf55c5666))
* lr multipliers and make group params not optional ([b4b9fdf](https://github.com/rlcoretech/core-rl/commit/b4b9fdffba6a718219668c4b2a27de06aa91b5c0))
* Make BaseEventBus take a generic EventTopic ([50fb186](https://github.com/rlcoretech/core-rl/commit/50fb186da8622395abc5ac8fbde6b42e4b7db500))
* Minor change in readme ([f6973e8](https://github.com/rlcoretech/core-rl/commit/f6973e8194d3457a9bde5d2c9bf87041174e4478))
* more styles ([4567c34](https://github.com/rlcoretech/core-rl/commit/4567c342af32bce0965e20650688243890dffc7b))
* more styles ([63ad85a](https://github.com/rlcoretech/core-rl/commit/63ad85a401b22a439652f110171ae720cc0fae28))
* more styles ([02f8e6a](https://github.com/rlcoretech/core-rl/commit/02f8e6a81d37e3988fa4ffb7738eca497596c09d))
* more styles and github workflow fixes ([63ad85a](https://github.com/rlcoretech/core-rl/commit/63ad85a401b22a439652f110171ae720cc0fae28))
* Move attach callback methods to BaseEventBus ([962e315](https://github.com/rlcoretech/core-rl/commit/962e315f3fe1155eb26df8ea4e6980643a1f0b00))
* Move scheduler clock to lib_utils ([62e6c97](https://github.com/rlcoretech/core-rl/commit/62e6c978c68db6e6daf077442909a1740dd68bd2))
* moving lib_utils message utils into own folder ([793c378](https://github.com/rlcoretech/core-rl/commit/793c3789d198ae78fbe28884fa0e3410bb7ea559))
* one last styles issue ([8c9d2ea](https://github.com/rlcoretech/core-rl/commit/8c9d2ea576169b0eb89e3d4b5eac51994d1b072c))
* one more style ([8c9d2ea](https://github.com/rlcoretech/core-rl/commit/8c9d2ea576169b0eb89e3d4b5eac51994d1b072c))
* **PROD-341:** prefer immutable/covariant input type annotations ([a3f8c87](https://github.com/rlcoretech/core-rl/commit/a3f8c87e7b7fdfd2d266792308e7950e0dc9ff40))
* **PROD-580:** use log probs instead of probs to compute actor loss for metrics ([2ec32e0](https://github.com/rlcoretech/core-rl/commit/2ec32e05d60fe0511c9a9c767b2a41d821c855b8))
* **PROD-580:** use log probs instead of probs to compute actor loss for metrics ([a3cfb98](https://github.com/rlcoretech/core-rl/commit/a3cfb9802232220e6ff470d59cc9c5de7d71fb55))
* **PROD-584:** Touching up logging messages and ZMQ Cleanup ([8e56bba](https://github.com/rlcoretech/core-rl/commit/8e56bba13c9a92a04d68e678a25c7321082ca056))
* **PROD-586:** ae cannot impute if too many traces are nan ([9a02cfa](https://github.com/rlcoretech/core-rl/commit/9a02cfa6dc60745f255d152b49b90861278bd28f))
* **PROD-587:** discard transitions that have *any* nans ([0b30c8c](https://github.com/rlcoretech/core-rl/commit/0b30c8c4fc8b7002fc5a2bdc665f43b58f3e0164))
* **PROD-587:** discard transitions that have *any* nans ([543b92b](https://github.com/rlcoretech/core-rl/commit/543b92b85282e764d0ad0f5c36fb4608a874b08e))
* **PROD-676:** remove action specific tags from BasicTagConfig ([f0a0966](https://github.com/rlcoretech/core-rl/commit/f0a096662b01b9b3866dcb5199c7ee17d709fcc4))
* **PROD-676:** remove action specific tags from BasicTagConfig ([d5e3ef2](https://github.com/rlcoretech/core-rl/commit/d5e3ef213a51e9f7ea25e0de134d00e9a778b7b5))
* **PROD-681:** make DeltaTagConfigs OPCTags ([fb968ae](https://github.com/rlcoretech/core-rl/commit/fb968ae21f2c17e3ade1e131bf83fb9f1f3387db))
* **PROD-681:** make DeltaTagConfigs OPCTags ([7c5d72c](https://github.com/rlcoretech/core-rl/commit/7c5d72cc3c90ed43ea225aacadb876f78d806097))
* **PROD-682:** make SafetyZonedTag bounds funcs and bounds tags populated by default ([a3f9425](https://github.com/rlcoretech/core-rl/commit/a3f942515f52ce9035024a43e82cd590f5e4f984))
* **PROD-682:** Populate SafetyZonedTag Bound Funcs and Bound Tags By Default ([dc36d72](https://github.com/rlcoretech/core-rl/commit/dc36d7225259d4bd9cbc8600960d65d5a542f07c))
* **PROD-712:** remove future annotations ([c7f3a13](https://github.com/rlcoretech/core-rl/commit/c7f3a13686473dd9d2df73b2696bbc924681ba4f))
* **PROD-713:** fix sync cond instantiation with group ([fc340c0](https://github.com/rlcoretech/core-rl/commit/fc340c05b91ac5e4306a54704379b156700fdb50))
* **PROD-735:** explicitly cast to float32 in allthetime tc ([69d1873](https://github.com/rlcoretech/core-rl/commit/69d18736b681cf93f89959a673f1339b1779c8d5))
* **PROD-738:** Event Bus Clock now in lib utils ([e4f5401](https://github.com/rlcoretech/core-rl/commit/e4f54016bf747b8f1fc26ea0e4a5100bbb62bae1))
* put config in corerl that wraps buffer config ([7379005](https://github.com/rlcoretech/core-rl/commit/73790058e6e4c88b9cbc0a51579f1a6f150a4586))
* refactor consumer threads ([c8e31a4](https://github.com/rlcoretech/core-rl/commit/c8e31a4772ae7878a8d7772f49704e3427619647))
* refactor corerl to use generic event bus ([7f8216b](https://github.com/rlcoretech/core-rl/commit/7f8216be1b252111f7e94cc7aef770bdbf4b2e4a))
* refactor event bus class to be generic ([3606cac](https://github.com/rlcoretech/core-rl/commit/3606cacda3818136cb3222f01f3ed6eda7cd4505))
* refactor event bus consumer threads ([60a45b6](https://github.com/rlcoretech/core-rl/commit/60a45b6a54e3b59e7cd2c2031ad7bf1d0a82cd25))
* refactor event types to be generic ([94fc7f7](https://github.com/rlcoretech/core-rl/commit/94fc7f76976a57d7524d5b8608ec929c87467e59))
* refactor event types to be generic ([9d80d0b](https://github.com/rlcoretech/core-rl/commit/9d80d0bd3d8fe1d2cf531590cd7e898db1c6032d))
* remove lower watermark configuration from database table fixtures ([2b620c6](https://github.com/rlcoretech/core-rl/commit/2b620c6acda65fa1ba4a7bb528c1706f422b5271))
* reward handling ([77ae304](https://github.com/rlcoretech/core-rl/commit/77ae3043a3a7299635b1eabbed9edb898981e657))
* specified jax transition type in greedyac ([02f8e6a](https://github.com/rlcoretech/core-rl/commit/02f8e6a81d37e3988fa4ffb7738eca497596c09d))
* styles ([8c9d2ea](https://github.com/rlcoretech/core-rl/commit/8c9d2ea576169b0eb89e3d4b5eac51994d1b072c))
* styles ([8c9d2ea](https://github.com/rlcoretech/core-rl/commit/8c9d2ea576169b0eb89e3d4b5eac51994d1b072c))
* styles ([7975f1d](https://github.com/rlcoretech/core-rl/commit/7975f1d80058ecd38de6f95f1ab3e9a7677d35ca))
* styles ([7975f1d](https://github.com/rlcoretech/core-rl/commit/7975f1d80058ecd38de6f95f1ab3e9a7677d35ca))
* styles ([4567c34](https://github.com/rlcoretech/core-rl/commit/4567c342af32bce0965e20650688243890dffc7b))
* styles ([4567c34](https://github.com/rlcoretech/core-rl/commit/4567c342af32bce0965e20650688243890dffc7b))
* styles ([4567c34](https://github.com/rlcoretech/core-rl/commit/4567c342af32bce0965e20650688243890dffc7b))
* styles ([14d4c77](https://github.com/rlcoretech/core-rl/commit/14d4c773a7e34e1daf736ecbbe733067e4eb92e2))
* styles ([5a08d17](https://github.com/rlcoretech/core-rl/commit/5a08d17555be92c7c43fc79c0c1b8d15ae90d48c))
* styles and import issues ([02f8e6a](https://github.com/rlcoretech/core-rl/commit/02f8e6a81d37e3988fa4ffb7738eca497596c09d))
* styles and move datamode ([02f8e6a](https://github.com/rlcoretech/core-rl/commit/02f8e6a81d37e3988fa4ffb7738eca497596c09d))
* styles and transition convertion ([02f8e6a](https://github.com/rlcoretech/core-rl/commit/02f8e6a81d37e3988fa4ffb7738eca497596c09d))
* styles and uv freeze in github workflow ([63ad85a](https://github.com/rlcoretech/core-rl/commit/63ad85a401b22a439652f110171ae720cc0fae28))
* the pyright problem ([744fa9d](https://github.com/rlcoretech/core-rl/commit/744fa9dfcd9c1b32faa064e08e24abed23addc21))
* transition type conversion, buffers now accept named tuples instead ([02f8e6a](https://github.com/rlcoretech/core-rl/commit/02f8e6a81d37e3988fa4ffb7738eca497596c09d))
* update log message for hard sync condition in BufferedWriter ([055fa2d](https://github.com/rlcoretech/core-rl/commit/055fa2def625b8bc67250a96d4e83713434fee54))
* update scheduler docstring ([27c4808](https://github.com/rlcoretech/core-rl/commit/27c4808c399431ebc50e8345a29c7b5c7764291b))
* update uv ([1a6814a](https://github.com/rlcoretech/core-rl/commit/1a6814abb9c8b1eec5e2fc595976e9ca56f2f353))
* update watermark configuration to include name in EvalDBConfig and MetricsDBConfig ([30d4291](https://github.com/rlcoretech/core-rl/commit/30d4291817a3233f9d72fb45629e82406d518fd8))
* updated config files that violated max_n_step &gt; steps_per_decision ([3a83ba8](https://github.com/rlcoretech/core-rl/commit/3a83ba881e698c495b9510b32258bba329793b30))
* using new Generic syntax ([8d35c0c](https://github.com/rlcoretech/core-rl/commit/8d35c0c100c70d774f5c9a26b22812e5407427f3))
* **workflows, corerl:** remove corerl build docker test ([4da3750](https://github.com/rlcoretech/core-rl/commit/4da375040fa482c2133967bf44caba7c1d55edeb))
* wrap coreio loop in try/except ([bceb18f](https://github.com/rlcoretech/core-rl/commit/bceb18fa45f40234c3fd7386edf59afb0192fc6f))
* wrap coreio loop in try/except ([f0181ea](https://github.com/rlcoretech/core-rl/commit/f0181ea20bc8376b3cb0dc3997fab20899c4557a))
