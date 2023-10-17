# TMP
TMP: Task-oriented Memory-efficient Pruning-adapter

- calculate_importance.py calculates the importance of all tasks to the Transformer model structure and places visual charts and importance npy files in the importance folder
- original.py trains and tests the performance of traditional bert methods on GLUE datasets
- Lora.py trains and tests the performance of traditional LoRA methods on GLUE datasets
- Pruning-Adapter.py trains and tests the performance of the article's method on the GLUE dataset
- trainable_params.py calculates model parameters (total parameters, trainable parameters)
- GPUMem_train.py calculates the GPU usage of the model
- evaluate.py only tests the performance of individual methods on the GLUE dataset (select methods by comments)



Save visual charts and importance .npy files in the Importance folder
