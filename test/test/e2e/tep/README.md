## Wrap Fortran code in Python using f2py (following the smart way)

See [this](https://docs.scipy.org/doc/numpy/f2py/getting-started.html#the-smart-way) for more details.

1. Create a signature file from fortran source code by running:

    ```
    uv run python -m numpy.f2py src/temain_mod.f src/teprob.f -m temain_mod -h temain_mod-auto.pyf
    ```

2. Update the intent of the arguments of the target functions (use `intent(in)` and `intent(out)` attribute) . You should do this by editing the signature file `temain_mod-auto.pyf`.

    The final version is:
    
    ```
    subroutine step(xdata,actions,idata) ! in :temain_mod:src/temain_mod.f:temain:unknown_interface
        double precision dimension(53), intent(in) :: xdata
        double precision dimension(13), intent(out) :: actions
        integer dimension(20), intent(out) :: idata
    ```

    Save this as temain_mod-smart.pyf

3. Build the extension module by running:

    ```
    uv run python -m numpy.f2py -c temain_mod-smart.pyf src/temain_mod.f src/teprob.f
    ```

4. Import the module in python:

    ```
    import temain_mod
    ```

5. To run the process simulation with an OPC interface:
    First bring up the required docker containers:
    ```
    docker compose up telegraf grafana -d
    ```
    Then from /test/test/e2e/tep start the simulation by calling:
    ```
    uv run python opc_tep.py --config-name ../../../../config/dep_tennessee_eastman_process.yaml
    ```
    Then bring up coreio, by navigating to /coreio:
    ```
    uv run python coreio/main.py --config-name ../config/dep_tennessee_eastman_process.yaml
    ```
    Next start the agent from /corerl:
    ```
    uv run python corerl/main.py --config-name ../config/dep_tennessee_eastman_process.yaml
    ```
    Finally launch an opcua-client to interact with the simulation. cd to /coreio and run
    ```
    uv pip install opcua-client pyqtgraph
    uv run opcua-client
    ```