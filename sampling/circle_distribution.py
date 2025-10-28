import torch
import math
from torch.distributions import Distribution, constraints
from torch.distributions.utils import broadcast_all


class UniformDisk(Distribution):
    """
    This class defines a uniform probability distribution over a 2D disk.
    By inheriting from `torch.distributions.Distribution`, we get a standard structure
    that allows us to use our custom class just like a native PyTorch distribution,
    with standard methods like `.sample()` and `.log_prob()`.
    """
    # `arg_constraints` is a dictionary that tells PyTorch about the constraints on the
    # constructor's parameters. Here, we specify that the 'radius' parameter must be a positive number.
    # PyTorch can use this information to automatically validate user inputs if `validate_args=True`.
    arg_constraints = {'radius': constraints.positive}

    # `has_rsample` is a boolean flag indicating whether the distribution supports the "reparameterization trick".
    # This trick is an advanced technique that allows gradients to flow back through the sampling
    # operation, which is essential for training models like Variational Autoencoders (VAEs).
    # Since our polar coordinate sampling method involves a `sqrt` of a random variable,
    # it is not straightforwardly reparameterizable, so we disable this feature.
    has_rsample = False

    def __init__(self, radius=1.0, validate_args=None):
        # This is the constructor for our class.
        # `broadcast_all` is a PyTorch utility that handles parameter "broadcasting".
        # It allows us to create a batch of distributions at once. For example, if `radius` is a tensor
        # of 5 values, `broadcast_all` ensures all parameters have the correct shape to create 5 parallel distributions.
        # The comma in `self.radius,` is a Python trick to unpack the result (which is a tuple) into a single variable.
        self.radius, = broadcast_all(radius)

        # We must call the constructor of the parent `Distribution` class. This is a required step.
        # `batch_shape`: Defines how many "parallel" distributions we are managing. If `self.radius` has a shape of (5,),
        #                then `batch_shape` will be (5,), indicating a batch of 5 distributions, each with a different radius.
        # `event_shape`: Defines the shape of a single sample (an "event") from the distribution. For us,
        #                a single sample is a 2D point (x, y), so its shape is `torch.Size([2])`.
        super().__init__(batch_shape=self.radius.shape, event_shape=torch.Size([2]), validate_args=validate_args)

    def sample(self, sample_shape=torch.Size()):
        """
        This method generates samples from our distribution.
        It implements the Inverse Transform Sampling method using polar coordinates,
        which is 100% efficient (no samples are rejected).
        """
        # `_extended_shape` is a helper function from the base class that computes the final shape of the output tensor.
        # It combines the requested `sample_shape` (e.g., (100,)) with the distribution's `batch_shape`.
        shape = self._extended_shape(sample_shape)
        # `shape.numel()` calculates the total number of scalar elements in the final output tensor.
        # Since each of our samples is a 2D point, we divide by 2 to get the number of (x,y) points to generate.
        n_samples = shape.numel() // 2

        # --- Step 1: Sample the angle `theta` ---
        # For a uniform distribution on a disk, there is no preferred direction.
        # Therefore, we sample the angle `theta` from a Uniform distribution over the interval [0, 2*pi].
        theta = torch.distributions.Uniform(low=0.0, high=2 * math.pi).sample((n_samples,))

        # --- Step 2: Sample the radius `r` ---
        # This is the key insight. Simply sampling `r` uniformly from [0, R] would incorrectly
        # cluster points near the center. To achieve a uniform *spatial* density, the probability
        # of sampling a certain radius must be proportional to the radius itself (the larger the radius,
        # the larger the circumference, so more points should land there).
        # The resulting PDF for the radius is p(r) = 2r/R^2. Its CDF is F(r) = (r/R)^2.
        # To invert the CDF, we set u = (r/R)^2 (where u is a sample from U(0,1)) and solve for r: r = R * sqrt(u).

        # Sample `n_samples` values from a Uniform(0, 1) distribution.
        u = torch.rand(n_samples)
        # Apply the inverse CDF formula to get the correctly distributed radii.
        r = self.radius * torch.sqrt(u)

        # --- Step 3: Convert from polar to Cartesian coordinates ---
        # We use the standard conversion formulas: x = r*cos(theta), y = r*sin(theta).
        x = r * torch.cos(theta)
        y = r * torch.sin(theta)

        # `torch.stack` combines the `x` and `y` tensors into a single tensor.
        # `dim=-1` ensures that for each index `i`, `samples[i]` becomes the pair `[x[i], y[i]]`.
        # The result is a tensor of shape (n_samples, 2).
        samples = torch.stack([x, y], dim=-1)
        # Finally, we reshape the tensor of samples to the final required shape (calculated at the beginning).
        return samples.reshape(shape)

    def log_prob(self, value):
        """
        This method calculates the logarithm of the probability density (log PDF) for a given point `value`.
        `value` is a tensor of (x, y) points for which we want to know the density.
        """
        # If `validate_args` is True, this optional check verifies that the shape of `value`
        # is compatible with the distribution's `event_shape`.
        if self._validate_args:
            self._validate_sample(value)

        # --- Density Calculation ---
        # For a uniform distribution, the probability density is constant inside its domain (support)
        # and zero outside. The value of the density is `1 / Area_of_the_domain`.
        # The area of a disk is pi * r^2.
        area = math.pi * self.radius ** 2
        # Working with logarithms is more numerically stable. The log-density is log(1/Area) = -log(Area).
        log_p = -torch.log(area)

        # To check if a point is inside or outside the circle, we calculate its squared distance from the origin.
        # This is more efficient than calculating the actual distance because it avoids a square root operation.
        # `value` has a shape of (..., 2). `value**2` squares both x and y coordinates.
        # `torch.sum(..., dim=-1)` sums x^2 and y^2 along the last dimension to get the squared distance.
        distance_sq = torch.sum(value ** 2, dim=-1)

        # `torch.where` is a vectorized conditional operation. It works like an "if/else" on every element of a tensor.
        # For each point, it checks if `distance_sq <= self.radius**2`.
        # - If the condition is True (the point is inside or on the edge of the circle), it returns `log_p`.
        # - If the condition is False (the point is outside), it returns `-inf` (which is the logarithm of zero).
        return torch.where(distance_sq <= self.radius ** 2, log_p, -float('inf'))
