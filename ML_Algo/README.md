Part 2a: Sync gradient with gather and scatter call using Gloo backend
PyTorch comes with the Gloo backend built-in. We will use this to implement gradient aggregation using the gather and scatter calls.

(i) Set Pytorch up in distributed mode using the distributed module of PyTorch. For details look here. Initialize the distributed environment using init_process_group.

(ii) Next, to perform gradient aggregation you will need to read the gradients after backward pass for each layer. Pytorch performs gradient computation using auto grad when you call .backward on a computation graph. The gradient is stored in .grad attribute of the parameters. The parameters can be accessed using model.parameters().

(iii) Finally, to perform the aggregation you will you use gather and scatter communication collectives. Specifically Rank (worker) 0 in the group will gather the gradients from all the participating workers and perform elementwise mean and then scatter the mean vector. The workers update the grad variable with the received vector and then continue training.

Task2a: Implement gradient sync using gather and scatter. Verify that you are using the same total batch size, where total batch size = batch size on one machine * num_of machines. With the same total batch size you should get similar loss values as in the single node training case. Remember you trained with total batch size of 256 in Task 1.

Part 2b: Sync gradient with allreduce using Gloo backend
Ring Reduce is an extremely scalable technique for performing gradient synchronization. Read here about how ring reduce has been applied to distributed machine learning. Instead of using the scatter and gather collectives separately, you will next use the built in allreduce collective to sync gradients among different nodes. Again read the gradients after backward pass layer by layer and perform allreduce on the gradient of each layer. Note the PyTorch allreduce call doesn’t have an ‘average’ mode. You can use the ‘sum’ operation and then get the average on each node by dividing with number of workers participating. After averaging, update the gradients of the model as in the previous part.

Task2b: Implement gradient sync using allreduce collective in Gloo. In this case if you have set the same random seed (see FAQ), you should see the same loss value as in Task2a, while using the same total batch size of 256.
