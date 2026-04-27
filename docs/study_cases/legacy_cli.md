# Legacy CLI Commands

> **Type:** Reference — lookup of legacy `emhass --action ...` command-line equivalents.

The study-case tutorials and how-to guides primarily show REST API calls (curl, `rest_command`, or the Add-on action button), because the EMHASS Add-on and standalone Docker are the recommended deployment targets and these expose REST.

This page collects the **legacy Python virtual-environment CLI** equivalents for users running EMHASS as a `pip`-installed Python module — for example, in Docker-standalone scenarios, CI pipelines, or backtest scripts.

## Day-ahead optimization

REST (recommended):

```bash
curl -i -H "Content-Type: application/json" \
     -X POST -d '{}' \
     http://localhost:5000/action/dayahead-optim
```

Legacy CLI:

```bash
emhass --action 'dayahead-optim' \
       --config '/home/user/emhass/config_emhass.json' \
       --costfun 'profit'
```

## Perfect optimization (7-day historical backtest)

REST:

```bash
curl -i -H "Content-Type: application/json" \
     -X POST -d '{}' \
     http://localhost:5000/action/perfect-optim
```

Legacy CLI:

```bash
emhass --action 'perfect-optim' \
       --config '/home/user/emhass/config_emhass.json' \
       --costfun 'profit'
```

## Naive MPC optimization

REST:

```bash
curl -i -H "Content-Type: application/json" \
     -X POST -d '{"prediction_horizon": 24}' \
     http://localhost:5000/action/naive-mpc-optim
```

Legacy CLI:

```bash
emhass --action 'naive-mpc-optim' \
       --config '/home/user/emhass/config_emhass.json' \
       --runtimeparams '{"prediction_horizon": 24}'
```

## Publish data

REST:

```bash
curl -i -H "Content-Type: application/json" \
     -X POST -d '{}' \
     http://localhost:5000/action/publish-data
```

Legacy CLI:

```bash
emhass --action 'publish-data' \
       --config '/home/user/emhass/config_emhass.json'
```

## Flag mapping

| REST body field | Legacy CLI flag |
|-----------------|-----------------|
| (request body, JSON) | `--runtimeparams '{...}'` |
| (`X-Cost-Function` header *not used*) | `--costfun 'profit' \| 'cost' \| 'self-consumption'` |
| (config-file via Add-on options or running `/get-config`) | `--config /path/to/config.json` |
| (logs in `/share/data/`) | `--data /path/to/data/` |

## When to use this surface

- **Docker-standalone** without the Home Assistant Add-on supervisor
- **CI pipelines** that need to run `perfect-optim` against historical data without a long-running server
- **Backtest scripts** that script multiple `--action` calls in sequence

For all other deployments — Add-on, standalone-with-supervisor, Home Assistant integration — prefer the REST surface.

## See also

- [Automations](../automations.md) — Reference: full `shell_command` and `rest_command` patterns
- [Passing data](../passing_data.md) — Reference: complete list of `runtimeparams` keys
- [Configuration](../config.md) — Reference: every parameter
