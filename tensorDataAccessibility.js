// how do we access elements inside a tensor?

let data = tf.tensor([10, 20, 30]);

// cannot do data[0]
data.get(0);

// for 2d:
data = tf.tensor([
  [10, 20, 30],
  [40, 50, 60]
])

// data.get(0) will return a rank error (row first, then col)
data.get(0, 0);
data.get(1, 2);

// contrary to intuition, you cannot set values in a tensor
// you would need to create a new tensor, but you can
// create a new tensor via tensor operations (they return
// tensors)

// looking back to our KNN algorithm, it was a bit awkward
// to retrieve values from a certain column, let's see
// how to do that in tf:

data = tf.tensor([
	[1, 2, 3],
  [4, 5, 6],
  [1, 2, 3],
  [4, 5, 6],
  [1, 2, 3],
  [4, 5, 6]
]);

// let's get the center column: slice needs two args,
// start index, and size. the start index are the coordinates
// of the starting element in [row, column] format. the size
// arg (not zero indexed) is how many rows we wanna slice, and
// how many columns we wanna take, so
// [no. of rows going down, no. of cols going right]

data.slice([0, 1], [6, 1]);

// what if you do not know how many rows to take? you
// can get the number of rows from data.shape()[0] but
// that's a little nasty- if you want all the rows below
// the value you can just go with a -1

data.slice([0, 1], [-1, 1]);
