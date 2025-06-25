## Backend system architecture diagram

- Functions with an `*` are completed to an MVP level
- `CoreUI` is not part of the backend, but is represented here to communicate its communication pathway into the backend.

```mermaid
flowchart TD
    CoreUI[CoreUI]

    CoreAuth["CoreAuth<br/> - user authentication<br/> - session management<br/> - access control<br/> - software license key<br/> - service endpoints"]

    Coredinator["Coredinator<br/> - spin up services*<br/> - multi-agent orchestration*<br/> - installation<br/> - healthcheck*<br/>"]

    CoreIO["CoreIO<br/> - data ingress<br/> - write setpoints to opc*<br/> - write alarms to opc<br/> - write heartbeat to opc*<br/> - high-freq events triggering<br/> - authenticated opc handshake*<br/> - encrypted comms*<br/> - outbound opc data validation"]

    CoreRL["CoreRL<br/> - agent updating*<br/> - setpoint recommendations*<br/> - internal health metrics*<br/> - estimated value add*<br/> - predicted next setpoint*<br/> - goal satisfaction metrics*<br/> - constraint satisfaction metrics*<br/> - optimization metrics*<br/> - uncertainty estimation"]

    SensorDB[(SensorDB*)]
    DataPipeline["DataPipeline<br/> - virtual tags*<br/> - anomaly detection<br/> - missing data imputation*<br/> - event triggering*<br/> - data preprocessing*<br/> - constraints encoding*<br/> - goal specification*"]
    Buffer[(Buffer)]

    CoreTelemetry["CoreTelemetry<br/> - cpu / ram / disk monitoring<br/> - local metrics caching*<br/> - cloud if outbound internet<br/> - error logging and observability<br/> - server instrumentation"]
    LocalCache[(LocalCache*)]
    CloudMetrics[(CloudMetrics)]

    CoreConfig["CoreConfig<br/> - validation*<br/> - live updating<br/> - config access control<br/> - audit logging"]
    ConfigDB[(ConfigDB)]

    CoreUI --> CoreAuth
    CoreAuth --> Coredinator

    subgraph AuthenticatedUser
        Coredinator --> CoreTelemetry
        Coredinator --> CoreIO
        Coredinator --> CoreRL
        Coredinator <--> CoreConfig

        subgraph Comms
            CoreTelemetry <--> LocalCache
            CoreTelemetry <--> CloudMetrics
            LocalCache -.-> CloudMetrics
            CoreIO --> SensorDB
        end


        SensorDB --> DataPipeline
        CoreConfig <--> ConfigDB
        CoreConfig --> CoreRL

        subgraph Agent
            DataPipeline --> Buffer
            Buffer --> CoreRL
        end
    end
```
