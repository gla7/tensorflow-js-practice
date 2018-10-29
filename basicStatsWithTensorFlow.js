const numbers = tf.tensor([
	[1, 2],
  [3, 4],
  [5, 6]
]);

// moments returns an object with statistical quantities
// like mean and variance, and its second argument tells
// the method whether to get the moments on the rows or
// the columns, so we need them for the columns
const { mean, variance } = tf.moments(numbers, 0);

// in order to standardize numbers we go (value - avg) / stdv
numbers.sub(mean).div(variance.pow(0.5))
