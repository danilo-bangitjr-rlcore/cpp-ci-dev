# Changelog

## [0.2.0](https://github.com/rlcoretech/core-rl/compare/coretelemetry-v0.1.0...coretelemetry-v0.2.0) (2025-11-05)


### Features

* Adding metrics filtering logic to backend ([c144492](https://github.com/rlcoretech/core-rl/commit/c144492fb0284c0675e9deb4a373af3d241a207c))


### Bug Fixes

* **CI:** CoreTelemetry flaky database creation ([78525e1](https://github.com/rlcoretech/core-rl/commit/78525e1b5e6baf6866510fff82cbd259bfffd5ae))
* **CI:** CoreTelemetry only comments coverage percentage when running on Linux ([a2d97ec](https://github.com/rlcoretech/core-rl/commit/a2d97ec11b81a33c8002b640d03963362fcdf234))
* **coretelemetry:** more retries in sql creation ([34b5f99](https://github.com/rlcoretech/core-rl/commit/34b5f9957214f6fd2e36be01010b2ce28f62847e))
* **CoreUI:** Refactor agent metrics so filtering happens in Backend ([85ceae7](https://github.com/rlcoretech/core-rl/commit/85ceae78f7c05db062c63310e958bc50323cf84a))

## [0.1.0](https://github.com/rlcoretech/core-rl/compare/coretelemetry-v0.0.1...coretelemetry-v0.1.0) (2025-10-29)


### Features

* add test db endpoint ([4c9899a](https://github.com/rlcoretech/core-rl/commit/4c9899a3ba60d743c140fc8509b27c7850cdfa16))
* **CoreGateway:** Talk to Coretelemetry (PROD-1133) ([6ef4d6d](https://github.com/rlcoretech/core-rl/commit/6ef4d6de04e170df74d79ca8f77314e564ff46be))
* coretelemetry first commit ([837ccd8](https://github.com/rlcoretech/core-rl/commit/837ccd8bc53a8701e98124d8e0215bdf2ea18ae6))
* **coretelemetry:** add API endpoints for telemetry data and configuration management ([450a232](https://github.com/rlcoretech/core-rl/commit/450a232ee2fc5d38c1cd484f92c8182d7e4843e6))
* **coretelemetry:** add clear cache endpoint ([93a9bb4](https://github.com/rlcoretech/core-rl/commit/93a9bb413292422b2b46053fb2d6fec7bb3f725f))
* **coretelemetry:** add CLI args for config path and port ([5b23f6a](https://github.com/rlcoretech/core-rl/commit/5b23f6aaad3c8fc5641c2a2d9fb118454175b80a))
* **coretelemetry:** add response models for API endpoints ([7b9f850](https://github.com/rlcoretech/core-rl/commit/7b9f850937518f9a0c714ca617cd7fc8ecdefcf8))
* **coretelemetry:** add services.py with TelemetryManager and DBConfig ([81ea9ad](https://github.com/rlcoretech/core-rl/commit/81ea9ad2aa54aa30d9accebb76a63fdafad413fd))
* **CoreTelemetry:** API for system metrics (PROD-1114) ([87cb1a2](https://github.com/rlcoretech/core-rl/commit/87cb1a28504d62b04fd7a4c3075ae5c0dbac368f))
* **coretelemetry:** First Commit ([dec8ec0](https://github.com/rlcoretech/core-rl/commit/dec8ec052f05e7b283b4c693f45c8051a868acd3))
* **coretelemetry:** first pass system metrics ([6419c83](https://github.com/rlcoretech/core-rl/commit/6419c833515a40380f7055d7e1504046bb9f18ef))
* **coretelemetry:** implement single column reader ([bcaaf0b](https://github.com/rlcoretech/core-rl/commit/bcaaf0b3c7c41a002ba08d88b8e027ec52afadd4))
* **coretelemetry:** more endpoints ([eda7ebc](https://github.com/rlcoretech/core-rl/commit/eda7ebcdff82aff0ce9c525875f2a953a7bc9a1e))
* **coretelemetry:** tests first commit ([0e50259](https://github.com/rlcoretech/core-rl/commit/0e502595b2db11bcba8e371a69d54c7006c43ea9))
* **coretelemetry:** update README ([24c2363](https://github.com/rlcoretech/core-rl/commit/24c23636b1618f76dffa4c86640f84718e9c0425))
* **coreui:** Per agent metrics [prod-1115] ([1bb0982](https://github.com/rlcoretech/core-rl/commit/1bb0982c1ca618251e8ab3635fd8ea1244941d95))
* **PROD-1111:** CoreTelemetry API ([6090e11](https://github.com/rlcoretech/core-rl/commit/6090e1146664384144727ef5c833a1a27a8d8f13))


### Bug Fixes

* Copilot's comments ([adec9fc](https://github.com/rlcoretech/core-rl/commit/adec9fc0b55165bcb4613cb84d3cfc11a9fa0844))
* **corer/coretelemetry:** only supports wide metrics ([419c5ea](https://github.com/rlcoretech/core-rl/commit/419c5ea1841fabb8d372928096022ff96e9fb9a0))
* **coretelemeetry:** config and reload ([b0eccc0](https://github.com/rlcoretech/core-rl/commit/b0eccc0056830b13ad473b36390eb96a7948e4df))
* coretelemetry can be installed with pyinstaller ([57569fa](https://github.com/rlcoretech/core-rl/commit/57569fa5a61e26485430aa5e8a4df3032f8dbe15))
* coretelemetry health to main ([c5a2b8d](https://github.com/rlcoretech/core-rl/commit/c5a2b8dbcc913d0a20ce1caeea2878de623bab4c))
* **coretelemetry:** API endpoint names ([69a8a00](https://github.com/rlcoretech/core-rl/commit/69a8a007bf555155b37c0b844feff7e8423dc113))
* **coretelemetry:** change default port ([eb9cf80](https://github.com/rlcoretech/core-rl/commit/eb9cf80e4f82cdd9403b30668d0f9b96813e1e58))
* **coretelemetry:** CI ([a3dd990](https://github.com/rlcoretech/core-rl/commit/a3dd990b1fedf62a9ad8158b0c21150cdfb91f9c))
* **coretelemetry:** cleanup comments ([8451a7b](https://github.com/rlcoretech/core-rl/commit/8451a7be370d89dc10858ca676b2b784b1be55d7))
* **coretelemetry:** db health ([9e01161](https://github.com/rlcoretech/core-rl/commit/9e011616725c6bd39be5bde2b014185328adc99b))
* **coretelemetry:** fixing data endpoint ([550e5f4](https://github.com/rlcoretech/core-rl/commit/550e5f4e383d53144734bb226d911301f201e363))
* **coretelemetry:** Installable with pyinstaller ([19c325a](https://github.com/rlcoretech/core-rl/commit/19c325a06c4d076f80668af100b3d534880b1b1b))
* **coretelemetry:** more refactor ([f57e0d0](https://github.com/rlcoretech/core-rl/commit/f57e0d0f5faea48d575314b9adb8bd822db96ce1))
* **coretelemetry:** move redirect to /docs from agent_metrics to app.py ([8368093](https://github.com/rlcoretech/core-rl/commit/8368093d4954056ed2d35dfc749496b655bf4493))
* **coretelemetry:** move SQL reader to own file ([8fe2906](https://github.com/rlcoretech/core-rl/commit/8fe2906e585021df1979c57f82488babe53a71fa))
* **coretelemetry:** Moving agent metrics into their own router ([0edb274](https://github.com/rlcoretech/core-rl/commit/0edb274f430dc90c9b35c7ed179da56350660d8c))
* **coretelemetry:** persistent config ([fce09bb](https://github.com/rlcoretech/core-rl/commit/fce09bbf80641e96b3813fc7b51a4240b885f6c9))
* **coretelemetry:** remove boilerplate from endpoints ([a0f3d15](https://github.com/rlcoretech/core-rl/commit/a0f3d1531c29b444bf00f7d77c27e64e111d5487))
* **coretelemetry:** rename metrics reader to sql reader ([486bb6a](https://github.com/rlcoretech/core-rl/commit/486bb6a7f684fff48f155d11d9034b1de948bac2))
* **coretelemetry:** return actual data and fix return types ([11cee52](https://github.com/rlcoretech/core-rl/commit/11cee521548e506ab04ea41fdded3f279f322131))
* **coretelemetry:** review large tests ([cc9e7cd](https://github.com/rlcoretech/core-rl/commit/cc9e7cd06aef213d17d08289835c383a9160ba7c))
* **coretelemetry:** review medium tests ([fbd3a53](https://github.com/rlcoretech/core-rl/commit/fbd3a53feff82efca33045c0199bf89ca3507e84))
* **coretelemetry:** review small tests ([cbf1586](https://github.com/rlcoretech/core-rl/commit/cbf1586542c61e68a5afb254b7a5c4f6ed341559))
* **coretelemetry:** separation of concerns in HTTP exceptions ([db17d73](https://github.com/rlcoretech/core-rl/commit/db17d7398cbe1dc3005a9bddcfb420c0813b6250))
* **coretelemetry:** supports watch reloads and executable ([459e66c](https://github.com/rlcoretech/core-rl/commit/459e66c6f0cede198bad8bfcb9c3077db23e8186))
* **coretelemetry:** system metrics api ([ff5351d](https://github.com/rlcoretech/core-rl/commit/ff5351d5eff482a2132a39bb76a723a484b17318))
* **coretelemetry:** TelemetryManager --&gt; AgentMetricsManager ([2e35b5c](https://github.com/rlcoretech/core-rl/commit/2e35b5ce375c3b1abf971d5e7d26a7f81f6d9c59))
* **coretelemetry:** update tests ([36fcc3d](https://github.com/rlcoretech/core-rl/commit/36fcc3d6dae7838bc4e2e366d2196812d4020bb9))
* endpoint documentation ([55163b0](https://github.com/rlcoretech/core-rl/commit/55163b0f49b57c05aab6d254cc9e01c1c8226460))
* **PROD-1188:** Coredinator, Coregateway, Coretelemetry reload fix ([fb2e453](https://github.com/rlcoretech/core-rl/commit/fb2e45352ce4301eb09d0c161db49bfd88ec0682))
* refactor for pyinstaller ([eba14f0](https://github.com/rlcoretech/core-rl/commit/eba14f046104138833afea873019bb158b5c0891))
* TelemetryException --&gt; AgentMetricsException ([d62832a](https://github.com/rlcoretech/core-rl/commit/d62832abc2408c5796f66fec79e5178e73847869))
