// concatenate two tensors:

let tensorA = tf.tensor([
	[1, 2, 3],
  [4, 5, 6]
]);

let tensorB = tf.tensor([
	[7, 8, 9],
  [10, 11, 12]
]);

// straight concatenation does not return the same shape
// i.e. from concating two [2, 3] we get a [4, 3]
tensorA.concat(tensorB);
tensorA.concat(tensorB).shape;

// we can add a second argument in concat which is the axis
// of concatenation, i.e. do we wish to concatenate the rows
// direction or do we wish to concatenate the columns direction
// (in a 2d example), so if we wanted to add columns:

tensorA.concat(tensorB, 1);
tensorA.concat(tensorB, 1).shape;
