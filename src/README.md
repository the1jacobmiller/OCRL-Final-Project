# Merge Controller Project

This package contains the code for our OCRL project here. The structure is as follows: 

* `merge_controller.ipynb` - The main experiment file
* `merge/sqp.py` - The main SQP loop
* `merge/lane_change_controller.py` - Controller that enacts a lane change
* `merge/nash_controller.py` - Controller that performs the optimization problem
* `merge/tests` - If you want to write tests they can go here, and can be run with `pytest` in the `OCRL-FINAL_PROJECT/src` directory. If you run `pytest` in the top level you will run all the flow tests...

# Merge Controller Dependencies

The specific OCRL project dependencies have been added inside the `merge.yml` file. Update the `flow` conda environment with the following command: 
```bash
conda env update --file merge.yml
```

