# Running the simulation

### from the main directory:
Run:
```
python -m hvbta.simulation.main_simulation
```

# Running CBS seperately

Run:
```
python3 Final_CBS.py input.yaml output.yaml
```
to run CBS on the map in input.yaml and store the schedules for all agents and all metrics in output.yaml

Run:
```
python3 visualize.py input.yaml output.yaml
```
to visualize the generated results 