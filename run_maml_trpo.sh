#!/bin/bash
# " faucet-open-v1" " basketball-v1" " faucet-close-v1" 
declare -a StringArray=("lever-pull-v1" "stick-push-v1" "handle-pull-side-v1" "stick-pull-v1" "dissassemble-v1" "coffee-push-v1" "hammer-v1" "plate-slide-side-v1" "handle-press-v1" "soccer-v1" "plate-slide-back-v1" "button-press-topdown-v1" "button-press-topdown-wall-v1" "peg-insert-side-v1" "push-wall-v1" "button-press-v1" "coffee-pull-v1" "window-close-v1" "door-open-v1" "drawer-open-v1" "box-close-v1" "door-unlock-v1")
for task in "${StringArray[@]}";
    do
    	echo "running $task"
    	python3 examples/simon_paper/maml_trpo_paper_ml1.py --name $task --seed 0
    done