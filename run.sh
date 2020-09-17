
#!/bin/bash

python experiment.py \
--training_data_directory ~/research/parkinsons_multimodal/organized_data \
--testing_data_directory ~/research/parkinsons_multimodal/organized_data \
--project_directory ~/research/parkinsons_multimodal/src/nmpevqvae/output/ \
--experiment_name Test_Run \
--device 1 \
--mode Inference \
--starting_iteration -1 \
--epochs 20000 \
--log_every 50 \
--checkpoint_every 200 \
--checkpoint_last 5 \
--batch_size 2 \
--learning_rate 0.0001 \
--loss Adaptive \
--reconstruction_lambda 1.0 \
--zero_image_gradient_loss 100000 \
--one_image_gradient_loss 10000 \
--max_image_gradient_loss 5 \
--first_decay_steps 6480 \
--alpha 0.0000001 \
--t_mul 1.25 \
--m_mul 0.95

