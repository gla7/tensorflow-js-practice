// tensorflow practice (tf)

const data = tf.tensor([1, 2, 3]);
const otherData = tf.tensor([4, 5, 6]);

data.shape
data.add(otherData); // like vector addition
data.sub(otherData); // like vector subtraction
data.mul(otherData); // like dot product
data.div(otherData); // like vector division

// higher dimensions:

const twoDData = tf.tensor([
  [1, 2, 3],
  [4, 5, 6]
]);
const otherTwoDData = tf.tensor([
  [4, 5, 6],
  [1, 2, 3]
]);

twoDData.shape
twoDData.add(otherTwoDData); // like vector addition
twoDData.sub(otherTwoDData); // like vector subtraction
twoDData.mul(otherTwoDData); // like dot product
twoDData.div(otherTwoDData); // like vector division

// when shapes do not match (broadcasting or smearing)
// we can still do tensor operations when, from right to left
// the shapes are the same OR one of them is equal to 1 OR
// in the dimension in question there is no corresponding value
// in one of the tensors

// here the shape of the first is [3] and the second is [1] so
// broadcasting is possible
const moreData = tf.tensor([1, 2, 3]);
const evenMoreData = tf.tensor([4]);

moreData.add(evenMoreData);

// here the shape of the first is [2, 3] and the second is
// [2, 1], so in the first dimension they match and the second
// we can apply the exception
const evenMoreDataTwoD = tf.tensor([
	[1],
  [1]
]);

twoDData.add(evenMoreDataTwoD);

// another example, suppose a shape of [2, 3, 2] vs [3, 1] then
// 2 3 2
//   3 1
// we can see that each of those complies with the rules, so
// operations are allowed

// another example, [2, 3, 2] vs [2, 1]
// 2 3 2
//   2 1
// not possible (3 vs 2 in middle column)

// when using tf, keep in mind that to see the tesors in a
// nice, arrayified way, you need to call e.g. data.print()
