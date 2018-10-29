// here we will implement a regression version of the knn
// algorithm using tf. basically our algorithm will largely
// stay the same as its classification counterpart. the
// main difference, of course, is that we are no longer
// doing a discrete prediction, but a continuous one.

// concretely, we are going to try and predict house prices
// given a dataset that has lattitude of house, longitude
// of house, and house price. the regression knn steps are:

// 1. have a desired latitude and longitude for a house
// 2. find the distance between every house in the data
//    and that house
// 3. sort the data by distance to desired house
// 4. take the top k results and average them

// as we can see, it is only step 4 that changes (and of
// course we are not yet accounting for optimizing k).

// another difference is that before we kept the label in
// the same dataset as the features for convenience using
// lodash. this is not best practice in tf, so we will
// separate label and features.

const features = tf.tensor([
	[-121, 47],
  [-121.2, 46.5],
  [-122, 46.4],
  [-120.9, 46.7]
]);

const labels = tf.tensor([
	[200],
  [250],
  [215],
  [240],
]);

// we want to predict price for these coordinates:
const predictionPoint = tf.tensor([
	[-121, 47]
]);

// we know pythagorean dist is sqrt((x - x')^2 + (y - y')^2 + ...)
// so we first subtract the prediction point on each feature,
// square it, add them, and then take the square root:
const distances = features
  .sub(predictionPoint)
  .pow(2) // square all values
	.sum(1) // sums the columns to each other
	.pow(0.5); // square roots all values

//distances; // [0, 0.5385153, 1.1661897, 0.3162265]
//distances.shape; // [4]

// the next step is to order from closest house to most
// distant house. thereir lies a complication; because
// features are separate to labels, if we change the
// order of the distances, these will no longer correspond
// to where their corresponding labels are, so we remedy
// with expanding dimension by the column and concat in
// the direction of column:

//distances.expandDims(1).concat(labels, 1); // [[0 , 200], [0.5385153, 250], [1.1661897, 215], [0.3162265, 240]]

// now, the next step is to order from lowest to greatest
// distance. before we did that via sorting arrays. here
// tf cannot sort tensors, so how do we do it? we could
// use the unstack method, which, for a 2d tensor, it
// takes each row and creates a new tensor out of each row
// which are gathered in a js array, which we can sort by
// the distance criterion:

// arbitrary k
const k = 2

let prediction = distances
  .expandDims(1)
  .concat(labels, 1)
  .unstack() // this is now a js array of tf tensors
	.sort((a, b) => a.get(0) > b.get(0) ? 1 : -1) // we sort by 0, which is the distance
	.slice(0, k) // take the top k records
	.reduce((acc, pair) => {
		return acc + pair.get(1); // sum all prices into acc
	}, 0)/k // divide by k to get average price

//prediction // 220

// so in summary, our prediction could have just been this
// code:

prediction = features
  .sub(predictionPoint) // subtract each feature from prediction point
  .pow(2) // square all values
	.sum(1) // sums the columns to each other
	.pow(0.5) // square roots all values
	.expandDims(1) // match the dimensions along column to match labels tensor
  .concat(labels, 1) // join the distances with prices
  .unstack() // this is now a js array of tf tensors
	.sort((a, b) => a.get(0) > b.get(0) ? 1 : -1) // we sort by 0, which is the distance
	.slice(0, k) // take the top k records
	.reduce((acc, pair) => {
		return acc + pair.get(1); // sum all prices into acc
	}, 0)/k; // divide by k to get average price
