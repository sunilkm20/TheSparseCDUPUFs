import numpy as np
# You are not allowed to use any ML libraries e.g. sklearn, scipy, keras, tensorflow etc

# SUBMIT YOUR CODE AS A SINGLE PYTHON (.PY) FILE INSIDE A ZIP ARCHIVE
# THE NAME OF THE PYTHON FILE MUST BE submit.py
# DO NOT INCLUDE OTHER PACKAGES LIKE SKLEARN, SCIPY, KERAS,TENSORFLOW ETC IN YOUR CODE
# THE USE OF ANY MACHINE LEARNING LIBRARIES WILL RESULT IN A STRAIGHT ZERO

# DO NOT CHANGE THE NAME OF THE METHOD my_fit BELOW
# THESE WILL BE INVOKED BY THE EVALUATION SCRIPT. CHANGING THESE NAMES WILL CAUSE EVALUATION FAILURE

# You may define any new functions, variables, classes here
# For example, functions to calculate next coordinate or step length

def Proximal_Operator(w, alpha, step_size):
    return np.maximum(0, np.abs(w) - alpha * step_size)

def acceleration(t,w,w_prev):
    t_prev = t
    t = (1 + np.sqrt(1 + 4 * t_prev**2)) / 2

    z = w + (t_prev - 1) / t * (w - w_prev)
    return (z,t)

# def convergence(X,w,y):
#     return np.sum((np.dot(X,w) - y) ** 2) / (2 *len(X) )

def Proximal_Gradient_Descent(X, y, alpha, step_size, max_iterations,tol):

    N,D = X.shape
    w = np.linalg.lstsq(X,y,rcond=None)[0]    # Initialising Weights with least square values.
    z = np.linalg.lstsq(X,y,rcond=None)[0]    
    t = 1
    mse_prev = 0

    for _ in range(max_iterations):
        w_prev = w.copy()
        z_prev = z.copy()

        # Gradient update
        residuals = y - np.dot(X, z)
        grad = -np.dot(X.T, residuals) / N
        w = Proximal_Operator(z - step_size * grad, alpha * step_size, step_size)
        
        # mse = convergence(X,w,y)
        # # Check for convergence
        # if(abs(mse - mse_prev)/mse * 100 < tol ):
        #     break  

        # Acceleration step
        z,t = acceleration(t,w,w_prev)

        # # Updating Mse
        # mse_prev = mse

    return w

################################
# Non Editable Region Starting #
################################
def my_fit( X_trn, y_trn ):
################################
#  Non Editable Region Ending  #
################################
   
    
    # Regularization constant(alpha)
    alpha = 4
    # Note - λ must be tuned as a hyperparameter but for the problem we have given you, 
    # the optimal value of lambda is expected to be somewhere in the range [0.1, 5]
    
    # Step-length (step_size)
    step_size = 0.0005

    # Maximum Iteration(max_iterations)
    max_iterations = 5000 

    # Tolerance 
    tol = 1e-6

    # Proximal gradient descent: this is a more advanced (but also speedier) technique that you should attempt by consulting online resources or contacting the instructor.
    # For the LASSO problem, the proximal descent method yields a technique called Iterative soft-thresholding algorithm (ISTA)
    
    model = Proximal_Gradient_Descent( X_trn, y_trn , alpha , step_size , max_iterations , tol )

    # model is Parameter or weights vector which is trained using proximal gradient

    return model
	# Use this method to train your model using training CRPs
	# Your method should return a 2048-dimensional vector that is 512-sparse
	# No bias term allowed -- return just a single 2048-dim vector as output
	# If the vector your return is not 512-sparse, it will be sparsified using hard-thresholding
	
    # Return the trained model