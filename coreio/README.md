# CoreIO

CoreIO is async first


## Testing
### Pytest
The e2e test for integration with `core-rl` is here:
```
pytest test/large/smoke/test_dep_mountain_car_continuous.py 
```


### Manual e2e test:
For this you will need 4 terminals:

* Terminal 1: In the core-rl python env, start a minimal OPC Server with 
```
uaserver --populate
```

* Terminal 2:  Start a graphic OPC client (e.g. opcua-client) and focus on the node `"ns=2;i=2"`
```
 uv run opcua-client
```

* Terminal 3: Start core-io with 
```
uv run python coreio/main_io.py --config-name coreio_test_config.yaml
```

* Terminal 4: Start the dummy_agent to send ZMQ _write_ messages to core-io:
```
uv run python coreio/dummy_agent.py --config-name coreio-test-config.yaml 
```

**Expected result:**
You should see in the OPC Client GUI that the node  `"ns=2;i=2"` is changing.

### Diagram
```
          ┌────────┐                   
          │ config │                   
          └────┬───┘                   
    ┌──────────┴──────────────┐        
    ▼                         ▼
┌───────┐                ┌────────┐    
│ Agent ├───────────────►│ CoreIO │    
└───────┘  ZMQ Pub/Sub   └────────┘    
                              ▲
                              │ 
                              │ asyncua
                              ▼ 
                        ┌────────────┐
                        │ OPC Server │
                        └────────────┘
```
