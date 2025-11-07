# Changelog

## [0.1.2](https://github.com/rlcoretech/core-rl/compare/coreui-v0.1.1...coreui-v0.1.2) (2025-11-07)


### Bug Fixes

* copilot's comments ([fb14ac6](https://github.com/rlcoretech/core-rl/commit/fb14ac635304e90b02e429d60c98fd5eb626f0a8))
* **CoreUI, client/server:** `get_all_configs` fault tolerance [PROD-1251] ([a658c87](https://github.com/rlcoretech/core-rl/commit/a658c878ad75ffef459d403e971a9bee0784517f))
* **coreui, server:** filter out non-agent configs from get_all_configs endpoint ([639d821](https://github.com/rlcoretech/core-rl/commit/639d821d14331d64ff03dd75b66f5078358f8298))
* **coreui,client:** frontend doesn't crash when receiving yaml's that are not configs ([ab8a420](https://github.com/rlcoretech/core-rl/commit/ab8a42084d4120390910ee0739fe587b79bbc5d8))
* **coreui:** Display Metrics Table Unavailable [Prod 1252] ([df6b79f](https://github.com/rlcoretech/core-rl/commit/df6b79fa0f5fd4038cd797fa9feab2b13ef48ba1))
* improve metrics loading ([58a1063](https://github.com/rlcoretech/core-rl/commit/58a10632b00ec92591a8aa2f7f74c506341dfb16))
* **PROD-1240:** CoreUI unified system metrics endpoint ([342c030](https://github.com/rlcoretech/core-rl/commit/342c03003926aa14a30339cdcc22f0f5b5f410bb))
* **PROD-1248:** CoreUI deprecate `clean` and `raw` notion ([917ce11](https://github.com/rlcoretech/core-rl/commit/917ce119ae45af034b61e6d5227a1b5c2e66adbd))

## [0.1.1](https://github.com/rlcoretech/core-rl/compare/coreui-v0.1.0...coreui-v0.1.1) (2025-11-05)


### Bug Fixes

* add factory icon to opc navigation ([4447d2d](https://github.com/rlcoretech/core-rl/commit/4447d2dd9371251381f122bb70b5cf0d50c26c00))
* **coreui/client:** now base url comes from window.location.host ([f24d9be](https://github.com/rlcoretech/core-rl/commit/f24d9be523723c2c14bd1a486f9237e06c8a5d52))
* **coreui/server:** fix cors in coreui server ([7a6cd5a](https://github.com/rlcoretech/core-rl/commit/7a6cd5ae20894f0fd3adf231db0cf6b93c2211bc))
* **coreui:** opc navigation supports different types of node identifiers ([1ffced1](https://github.com/rlcoretech/core-rl/commit/1ffced18507597d1d07180ad56ad46806a85ce4c))
* **coreui:** Opc Navigation supports different types of Node Ids ([cb1b403](https://github.com/rlcoretech/core-rl/commit/cb1b403403f4579cccf6a2c330b5d8e84448af5c))
* **CoreUI:** Presentation Improvements ([95caf5c](https://github.com/rlcoretech/core-rl/commit/95caf5c557653179a823de39cac236d026412edc))
* **coreui:** redirect to /app ([776a601](https://github.com/rlcoretech/core-rl/commit/776a60172f6aeb23d972d5b2dc2e82568417cb97))
* **coreui:** redirect to /app ([7acbd31](https://github.com/rlcoretech/core-rl/commit/7acbd31e7da8890e85f92519c9baeea26815284b))
* **CoreUI:** Refactor agent metrics so filtering happens in Backend ([85ceae7](https://github.com/rlcoretech/core-rl/commit/85ceae78f7c05db062c63310e958bc50323cf84a))
* imports w/ ruff ([83070eb](https://github.com/rlcoretech/core-rl/commit/83070eb7ea7ce384ca9ba3558e7502c8dea5b7a2))
* mountain car to more legible name in config ([33e823e](https://github.com/rlcoretech/core-rl/commit/33e823e54e1714704bfbc45191de7b7323d24a03))
* Moving diagnostics to agents overview ([ee6397d](https://github.com/rlcoretech/core-rl/commit/ee6397de03144b928170a1f01ccf83add4fd79d2))
* observation tags to input tags ([490d0cf](https://github.com/rlcoretech/core-rl/commit/490d0cf104907400bbb55df387431d426b916e01))
* **PROD-1246:** CoreUI network fixes ([24b8d79](https://github.com/rlcoretech/core-rl/commit/24b8d79a404be3e0b5512a4268725a6cc756b9b4))
* Re-imagined home page ([d55722e](https://github.com/rlcoretech/core-rl/commit/d55722ed123355d2712872eea285d1eabf580de2))
* remove about file ([c599765](https://github.com/rlcoretech/core-rl/commit/c599765b8c7991d8cc0186f07572282b31019dd2))
* rename about to diagnostics ([bc094ee](https://github.com/rlcoretech/core-rl/commit/bc094ee5fb0f0b437b0be1b4955ae0b1a1841f4b))
* reward to objective ([ef86f73](https://github.com/rlcoretech/core-rl/commit/ef86f73dc3f74fe0759e5e12818158a6e4852ff7))
* using backend to filter metrics ([54a8c1e](https://github.com/rlcoretech/core-rl/commit/54a8c1ecdba9bbc8df4e4269dc801f424a2a7136))

## [0.1.0](https://github.com/rlcoretech/core-rl/compare/coreui-v0.0.1...coreui-v0.1.0) (2025-10-29)


### Features

* **coreui:** add new coredinator states and functional on/off ([e6f1f30](https://github.com/rlcoretech/core-rl/commit/e6f1f3087dcd6bff4aec58b8992f178beb039550))
* **coreui:** add option to use running coreio for initial state ([26cf63b](https://github.com/rlcoretech/core-rl/commit/26cf63be0ed5060f8772e72cb19715abb8c80f40))
* **coreui:** Display System Metrics [PROD-1116] ([b13a977](https://github.com/rlcoretech/core-rl/commit/b13a97721f11741d05e0a7ab3605c0ffa0700076))
* **coreui:** integrate coredinator for agent list view + controls ([c31e852](https://github.com/rlcoretech/core-rl/commit/c31e85269765359c420a7dc6af50645c28a024e1))
* **coreui:** per agent metrics (still some integration problems with coredinator) ([5754742](https://github.com/rlcoretech/core-rl/commit/575474294d2cc93aaeb40622efbce2bf4c4b03ee))
* **coreui:** Per agent metrics [prod-1115] ([1bb0982](https://github.com/rlcoretech/core-rl/commit/1bb0982c1ca618251e8ab3635fd8ea1244941d95))
* **coreui:** system metrics ([f2850fd](https://github.com/rlcoretech/core-rl/commit/f2850fdacaf3b5ddb004fc9d27277449317c4a37))


### Bug Fixes

* Agent Name breaks to newline if it is too long ([2126ff0](https://github.com/rlcoretech/core-rl/commit/2126ff0179722795fc7c44a72006ddb160fe9ede))
* **build.py:** include --reload option for microservices ([0c57e92](https://github.com/rlcoretech/core-rl/commit/0c57e92cc89455754cfb43b1063cce9b1721b3de))
* change dir to path in build script ([9b1a92b](https://github.com/rlcoretech/core-rl/commit/9b1a92bb26e49e0fc3258d41f0a4816621628322))
* Copilot's comments ([52895c3](https://github.com/rlcoretech/core-rl/commit/52895c34260a71ac0c0fe37b1b9319407b42b04f))
* Copilot's comments ([adec9fc](https://github.com/rlcoretech/core-rl/commit/adec9fc0b55165bcb4613cb84d3cfc11a9fa0844))
* **corerl/coreui:** new defaults for corerl/config:metrics-table ([093b9a3](https://github.com/rlcoretech/core-rl/commit/093b9a3f6062e99f58c78c806985322e3da1db2c))
* **coreui/build:** separate default dirs for coretelemetry and coredinator ([4f0bcd0](https://github.com/rlcoretech/core-rl/commit/4f0bcd0270aa0be6f482013e96a9e732580d0e67))
* **coreui/client:** format ([59645dd](https://github.com/rlcoretech/core-rl/commit/59645ddc9fe95e83a20cfde10a4fbdf452c36ea3))
* **coreui/components:** center warning icon ([8a84d3b](https://github.com/rlcoretech/core-rl/commit/8a84d3b2dccccb3f3c5f2611b2dab293060fc20c))
* **coreui/components:** extract svgs from agent card into components ([20d1b78](https://github.com/rlcoretech/core-rl/commit/20d1b7865b3ed56f660ac54946a42aff135382b9))
* **coreui:** add a initial start agent state ([c295fee](https://github.com/rlcoretech/core-rl/commit/c295fee094b8e78f0eb2345a2331ec14c5668850))
* **coreui:** add get config path endpoint ([57cbebe](https://github.com/rlcoretech/core-rl/commit/57cbebedc2146dcb81e657dd9e47a22c26fabe0d))
* **coreui:** added path command line argumments for microservices ([f71a79c](https://github.com/rlcoretech/core-rl/commit/f71a79c991a6acce7828cb7cae3ca5bbda605eff))
* **coreui:** change config api to use user defined config directory path ([b56dc95](https://github.com/rlcoretech/core-rl/commit/b56dc951142f133349c41316e7e4271e12573a17))
* **coreui:** fix remove dupe div ([ac08951](https://github.com/rlcoretech/core-rl/commit/ac08951e4b8a9dcccdb82f46dc71470390362d58))
* **coreui:** formatting ([64fc024](https://github.com/rlcoretech/core-rl/commit/64fc0244b3081be829487d960e500f75938b6cdb))
* **coreui:** integrate coredinator agent status to agent cards ([4f61e75](https://github.com/rlcoretech/core-rl/commit/4f61e75f78c3d60140f88f697cf693a0a840bcd6))
* **coreui:** integrate coredinator agent status to agent cards ([4338b52](https://github.com/rlcoretech/core-rl/commit/4338b52aaaf1554a5d2ba5026feec00d173f3a57))
* **coreui:** integrate coredinator io and agent status to agent details ([5a3deeb](https://github.com/rlcoretech/core-rl/commit/5a3deebfb99fe754b3cac268cd5a0dbc1a01f2a8))
* **coreui:** integrate coredinator io and agent status to agent details ([0cab15a](https://github.com/rlcoretech/core-rl/commit/0cab15a977b72d9bb28fc4a0db9fe6c75c2f753a))
* **coreui:** integrate coredinator start/stop for agent and io ([9e66ad4](https://github.com/rlcoretech/core-rl/commit/9e66ad48fed077f65b97d625123d7215ab73fcdd))
* **coreui:** remove coreui coredinator proxy ([e7f6cbc](https://github.com/rlcoretech/core-rl/commit/e7f6cbcb2f00be5d374ed140f261d75ab28cee8d))
* **coreui:** remove coreui coredinator proxy ([6dfbdaa](https://github.com/rlcoretech/core-rl/commit/6dfbdaab521012ceadd17d4d1248e9b223e281ca))
* **coreui:** truncate long agent names ([d6e1e44](https://github.com/rlcoretech/core-rl/commit/d6e1e4455bd52fe0fd73138586ca6e8849568d11))
* **coreui:** updating build script ([4e6a248](https://github.com/rlcoretech/core-rl/commit/4e6a248398d076138782f39f66dc3a1263274acc))
* format ([53d7738](https://github.com/rlcoretech/core-rl/commit/53d7738303403ff3e4d51fe52e68de880c9f671a))
* PR comments ([978b2fd](https://github.com/rlcoretech/core-rl/commit/978b2fd6feea4719b559cb200d57a23aeea1fcab))
* **PROD-1188:** Coredinator, Coregateway, Coretelemetry reload fix ([fb2e453](https://github.com/rlcoretech/core-rl/commit/fb2e45352ce4301eb09d0c161db49bfd88ec0682))
