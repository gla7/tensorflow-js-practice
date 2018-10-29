// summing values along an axis

// imagine olympic distance jumping- we are given for
// each athlete data for 3 jumps so our data looks more or
// less like 4 5 6
//           5 7 2
//           9 8 9
// where each row is an athlete and each column is the result
// of a jump attempt. we are also given the athlete number and
// their height for each athlete like 1 182
//                                    2 173
//                                    3 187
// and what we want is each player's data in one tensor

const jumpData = tf.tensor([
	[4, 5, 6],
  [5, 7, 2],
  [9, 8, 9]
]);

const playerData = tf.tensor([
	[1, 182],
  [2, 173],
  [3, 187]
]);

// sum takes the axis you want to sum along, because we
// want to sum the columns together, we choose 1

jumpData.sum(1);
jumpData.sum(1).shape; // as it stands, we cannot concat this to
                       // playerData

// to remedy the above, we can keep the dimension, by passing
// true to our second argument and make sure that we concat
// on the columns and not the rows

jumpData.sum(1, true).concat(playerData, 1);

// or if you want the sum at the end

playerData.concat(jumpData.sum(1, true), 1);

// there is another way too using expandDims, its argument
// being where you want the dimension to be expand it

jumpData.sum(1).expandDims(1).shape;
jumpData.sum(1).expandDims(1).concat(playerData, 1);
