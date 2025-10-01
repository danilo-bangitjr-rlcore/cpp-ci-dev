# Changelog

## [0.2.2](https://github.com/rlcoretech/core-rl/compare/coredinator-v0.2.1...coredinator-v0.2.2) (2025-10-01)


### Bug Fixes

* **coredinator:** cleanup stop_process ([f7e3421](https://github.com/rlcoretech/core-rl/commit/f7e3421f17ec31ff92666b180a4da45590253a4b))
* **coredinator:** ensure base_path exists ([b504751](https://github.com/rlcoretech/core-rl/commit/b50475114b269607df0afdf74d7c37e08358c340))
* **coredinator:** return process_id and service name from healthcheck ([57a5775](https://github.com/rlcoretech/core-rl/commit/57a577571387760c6f7df19b52a230d39154cb45))
* **coredinator:** significantly improve process handling and logging for windows ([4dc276f](https://github.com/rlcoretech/core-rl/commit/4dc276fbe8242ada36b497156275c8c8addfa64c))

## [0.2.1](https://github.com/rlcoretech/core-rl/compare/coredinator-v0.2.0...coredinator-v0.2.1) (2025-09-26)


### Bug Fixes

* **coredinator:** improve windows service daemon capabilities ([d19985e](https://github.com/rlcoretech/core-rl/commit/d19985ee3fa4009bd3eb4809d6605a24fe57384a))
* **coredinator:** remove circular import issues ([8dc0616](https://github.com/rlcoretech/core-rl/commit/8dc06164db63ca074aed255445f469d3944ca57e))
* **coredinator:** use more reliable windows detection ([b782263](https://github.com/rlcoretech/core-rl/commit/b782263eea43d8996598f25f1080667db7c6dc03))

## [0.2.0](https://github.com/rlcoretech/core-rl/compare/coredinator-v0.1.0...coredinator-v0.2.0) (2025-09-24)


### Features

* **coredinator:** add ability to reattach to running processes ([75a61aa](https://github.com/rlcoretech/core-rl/commit/75a61aaeab947477bcfb7ba09e1f80eff7d9bab2))
* **coredinator:** add ability to share coreio instance across services ([b7c49dc](https://github.com/rlcoretech/core-rl/commit/b7c49dc6a48a399660eb85607f6777f5070d7a4f))
* **coredinator:** add basic service executable discovery ([8900619](https://github.com/rlcoretech/core-rl/commit/8900619d93a9e837f523a169d5666be01cf9d287))
* **coredinator:** add coreio standalone management routes ([2f2eff7](https://github.com/rlcoretech/core-rl/commit/2f2eff79eac6abdb7a0178e1edd89e91a4179b40))
* **coredinator:** add healthcheck endpoint checking for service status ([28154f3](https://github.com/rlcoretech/core-rl/commit/28154f3894e5bb05060bdbb6e5eb86ae70875d4f))
* **coredinator:** add moderate structured logging coverage ([2cbd1d8](https://github.com/rlcoretech/core-rl/commit/2cbd1d88c7154e513329a4b197f979f12ec021d3))
* **coredinator:** add multiagent coordination ([a6bf172](https://github.com/rlcoretech/core-rl/commit/a6bf172df79e6df3b33c20bd6e368f08c6f9c9b2))
* **coredinator:** add optional ability to specify coredinator port ([11ac4cc](https://github.com/rlcoretech/core-rl/commit/11ac4cc4e19598a9302991f63180e92208b25d55))
* **coredinator:** add private router for starting TEP demo from coredinator ([8d259ba](https://github.com/rlcoretech/core-rl/commit/8d259ba52b881599981d37b59cd1705e23bc99ad))
* **coredinator:** add rough sketch of agent_process manager ([f502e84](https://github.com/rlcoretech/core-rl/commit/f502e849e08060e94389f34182a9519c602b7dbc))
* **coredinator:** agent run/stop state should persist across coredinator restarts ([6421614](https://github.com/rlcoretech/core-rl/commit/64216144749120d9f825332bdfacff83b86de121))
* **coredinator:** agent status should report status of child services ([c08bf48](https://github.com/rlcoretech/core-rl/commit/c08bf486d2e9fe717352be1658043f5c955e8c4f))
* **coredinator:** automatically restart degraded services ([b32900a](https://github.com/rlcoretech/core-rl/commit/b32900a77b85939d95752e66e63d5b256c6d0e70))
* **coredinator:** coredinator executable ([405f7ff](https://github.com/rlcoretech/core-rl/commit/405f7ffcd079781dcdf390077eeda1150163fec0))
* **coredinator:** coredinator executable ([55e19ca](https://github.com/rlcoretech/core-rl/commit/55e19cae40f94c5425726b28e39a019cec587da6))
* **coredinator:** make coredinator runnable directly with python ([d30c0d2](https://github.com/rlcoretech/core-rl/commit/d30c0d23d8c4e0b9a753ff76156e78583271c9cb))
* **coredinator:** only build a Service if does not already exist ([dd3073b](https://github.com/rlcoretech/core-rl/commit/dd3073b81f77a0d9cb4f4c55bd16de947c9696ab))
* **coredinator:** setup coredinator internal utility api ([2782b40](https://github.com/rlcoretech/core-rl/commit/2782b40e9b8a6dddc3b1c8c2f350ece5ed45a923))
* **coredinator:** wire shared coreio id throughout fastapi public interface ([05c5575](https://github.com/rlcoretech/core-rl/commit/05c5575d8e3123dc9b2192ca9fd64326286b8aaa))


### Bug Fixes

* **coredinator:** change default port to 7000 ([0fa8b64](https://github.com/rlcoretech/core-rl/commit/0fa8b6418183178b33b7720d6013200d66fda43f))
* **coredinator:** fix merge conflicts ([2780c9f](https://github.com/rlcoretech/core-rl/commit/2780c9ff868d58b8fbeb43e44328bdf8b261655e))
* **coredinator:** move ServiceStatus to shared types module ([02cc992](https://github.com/rlcoretech/core-rl/commit/02cc992b0230c714bb10e78fa6682d577eca7c2a))
* **coredinator:** remove version finding logic for now ([943488b](https://github.com/rlcoretech/core-rl/commit/943488b31eba03950919663c0f0a43d3209595af))
* **coredinator:** robustify and simplify kill process logic ([c377a2f](https://github.com/rlcoretech/core-rl/commit/c377a2fdb1a1581a04fbff8439595ce49ef83d90))
* **coredinator:** separate fastapi instance from main for tests ([0c8d99d](https://github.com/rlcoretech/core-rl/commit/0c8d99df8d786de09e755119abf5609b38cc945c))
* **coredinator:** start api path returns an agentid ([c364671](https://github.com/rlcoretech/core-rl/commit/c364671403d6102d2e7bc261240668e6f6675735))
* **coredinator:** use richer globbing pattern to find service executables ([bc1c727](https://github.com/rlcoretech/core-rl/commit/bc1c72744df3ce611c1407fe93d26682e7f0e5ab))


### Documentation

* **coredinator:** add a comprehensive readme ([8425d04](https://github.com/rlcoretech/core-rl/commit/8425d04e0b066175b922b65a67da04c2f02aeb8f))
* **coredinator:** add TEP demo to readme ([6e09fba](https://github.com/rlcoretech/core-rl/commit/6e09fba1adc18b1a3774c6bba1d1d71352f2906e))
* **coredinator:** document new coredinator features ([6e17d9a](https://github.com/rlcoretech/core-rl/commit/6e17d9a76827fafbb853962bcd07082c24a8e9d2))

## [0.1.0](https://github.com/rlcoretech/core-rl/compare/v0.0.1...v0.1.0) (2025-09-18)


### Features

* **coredinator:** add ability to reattach to running processes ([75a61aa](https://github.com/rlcoretech/core-rl/commit/75a61aaeab947477bcfb7ba09e1f80eff7d9bab2))
* **coredinator:** add ability to share coreio instance across services ([b7c49dc](https://github.com/rlcoretech/core-rl/commit/b7c49dc6a48a399660eb85607f6777f5070d7a4f))
* **coredinator:** add basic service executable discovery ([8900619](https://github.com/rlcoretech/core-rl/commit/8900619d93a9e837f523a169d5666be01cf9d287))
* **coredinator:** add coreio standalone management routes ([2f2eff7](https://github.com/rlcoretech/core-rl/commit/2f2eff79eac6abdb7a0178e1edd89e91a4179b40))
* **coredinator:** add healthcheck endpoint checking for service status ([28154f3](https://github.com/rlcoretech/core-rl/commit/28154f3894e5bb05060bdbb6e5eb86ae70875d4f))
* **coredinator:** add moderate structured logging coverage ([2cbd1d8](https://github.com/rlcoretech/core-rl/commit/2cbd1d88c7154e513329a4b197f979f12ec021d3))
* **coredinator:** add multiagent coordination ([a6bf172](https://github.com/rlcoretech/core-rl/commit/a6bf172df79e6df3b33c20bd6e368f08c6f9c9b2))
* **coredinator:** add optional ability to specify coredinator port ([11ac4cc](https://github.com/rlcoretech/core-rl/commit/11ac4cc4e19598a9302991f63180e92208b25d55))
* **coredinator:** add private router for starting TEP demo from coredinator ([8d259ba](https://github.com/rlcoretech/core-rl/commit/8d259ba52b881599981d37b59cd1705e23bc99ad))
* **coredinator:** add rough sketch of agent_process manager ([f502e84](https://github.com/rlcoretech/core-rl/commit/f502e849e08060e94389f34182a9519c602b7dbc))
* **coredinator:** agent run/stop state should persist across coredinator restarts ([6421614](https://github.com/rlcoretech/core-rl/commit/64216144749120d9f825332bdfacff83b86de121))
* **coredinator:** agent status should report status of child services ([c08bf48](https://github.com/rlcoretech/core-rl/commit/c08bf486d2e9fe717352be1658043f5c955e8c4f))
* **coredinator:** automatically restart degraded services ([b32900a](https://github.com/rlcoretech/core-rl/commit/b32900a77b85939d95752e66e63d5b256c6d0e70))
* **coredinator:** coredinator executable ([405f7ff](https://github.com/rlcoretech/core-rl/commit/405f7ffcd079781dcdf390077eeda1150163fec0))
* **coredinator:** coredinator executable ([55e19ca](https://github.com/rlcoretech/core-rl/commit/55e19cae40f94c5425726b28e39a019cec587da6))
* **coredinator:** make coredinator runnable directly with python ([d30c0d2](https://github.com/rlcoretech/core-rl/commit/d30c0d23d8c4e0b9a753ff76156e78583271c9cb))
* **coredinator:** only build a Service if does not already exist ([dd3073b](https://github.com/rlcoretech/core-rl/commit/dd3073b81f77a0d9cb4f4c55bd16de947c9696ab))
* **coredinator:** setup coredinator internal utility api ([2782b40](https://github.com/rlcoretech/core-rl/commit/2782b40e9b8a6dddc3b1c8c2f350ece5ed45a923))
* **coredinator:** wire shared coreio id throughout fastapi public interface ([05c5575](https://github.com/rlcoretech/core-rl/commit/05c5575d8e3123dc9b2192ca9fd64326286b8aaa))


### Bug Fixes

* **coredinator:** change default port to 7000 ([0fa8b64](https://github.com/rlcoretech/core-rl/commit/0fa8b6418183178b33b7720d6013200d66fda43f))
* **coredinator:** fix merge conflicts ([2780c9f](https://github.com/rlcoretech/core-rl/commit/2780c9ff868d58b8fbeb43e44328bdf8b261655e))
* **coredinator:** move ServiceStatus to shared types module ([02cc992](https://github.com/rlcoretech/core-rl/commit/02cc992b0230c714bb10e78fa6682d577eca7c2a))
* **coredinator:** remove version finding logic for now ([943488b](https://github.com/rlcoretech/core-rl/commit/943488b31eba03950919663c0f0a43d3209595af))
* **coredinator:** robustify and simplify kill process logic ([c377a2f](https://github.com/rlcoretech/core-rl/commit/c377a2fdb1a1581a04fbff8439595ce49ef83d90))
* **coredinator:** separate fastapi instance from main for tests ([0c8d99d](https://github.com/rlcoretech/core-rl/commit/0c8d99df8d786de09e755119abf5609b38cc945c))
* **coredinator:** start api path returns an agentid ([c364671](https://github.com/rlcoretech/core-rl/commit/c364671403d6102d2e7bc261240668e6f6675735))
* **coredinator:** use richer globbing pattern to find service executables ([bc1c727](https://github.com/rlcoretech/core-rl/commit/bc1c72744df3ce611c1407fe93d26682e7f0e5ab))


### Documentation

* **coredinator:** add a comprehensive readme ([8425d04](https://github.com/rlcoretech/core-rl/commit/8425d04e0b066175b922b65a67da04c2f02aeb8f))
* **coredinator:** add TEP demo to readme ([6e09fba](https://github.com/rlcoretech/core-rl/commit/6e09fba1adc18b1a3774c6bba1d1d71352f2906e))
* **coredinator:** document new coredinator features ([6e17d9a](https://github.com/rlcoretech/core-rl/commit/6e17d9a76827fafbb853962bcd07082c24a8e9d2))
