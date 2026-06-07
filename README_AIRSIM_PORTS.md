# Running multiple AirSim training jobs

The Python configs now contain an `[airsim]` section:

```ini
[airsim]
ip = 127.0.0.1
port = 41451
timeout_value = 10.0
```

This controls the **Python client** endpoint only.  For two simulator windows to
run at the same time, each AirSim/Unreal instance must also be started with a
matching server-side `settings.json` that uses a different `ApiServerPort`.
Changing only the training config is not enough because both Unreal instances
would still try to listen on the same default AirSim RPC port.

Example for the first simulator:

```json
{
  "SettingsVersion": 1.2,
  "SimMode": "ComputerVision",
  "ApiServerPort": 41451
}
```

Example for the second simulator:

```json
{
  "SettingsVersion": 1.2,
  "SimMode": "ComputerVision",
  "ApiServerPort": 41452
}
```

Then set the matching training configs, for example:

```ini
# configs/config_NH_center_SimpleMultirotor_3D.ini
[airsim]
ip = 127.0.0.1
port = 41451
timeout_value = 10.0

# configs/config_City_400_Multirotor_2D.ini
[airsim]
ip = 127.0.0.1
port = 41452
timeout_value = 10.0
```

If a process prints `Connecting to AirSim at 127.0.0.1:41452` and then fails or
hangs, the most likely cause is that the matching Unreal/AirSim instance was not
started with `ApiServerPort: 41452`.


## What if training appears to hang after `Environment: ...`?

After the environment prints its selected map/dynamics/perception, the dynamics
constructor opens the AirSim RPC connection.  The code now prints the exact
endpoint first, for example:

```text
Connecting to AirSim at 127.0.0.1:41452 (timeout=10.0s)
```

If you do not see this line, make sure you are running the latest code.  If you
do see it and then get a connection error, the configured `[airsim]` endpoint is
not reachable: start the matching Unreal/AirSim scene or fix the config port to
match that scene's `ApiServerPort`.
