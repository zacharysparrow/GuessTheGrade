# GuessTheGrade
*Public repository hidden at the request of Aurora Climbing Inc.*

Check out my <a href="zacharysparrow.github.io/projects/guess_the_grade">blog post</a>!

<div align="center">
  <a href="https://github.com/zacharysparrow/GuessTheGrade/">
    <img src="assets/climb_10_attention_weights.png" alt="TB Attention Weights" height="800">
  </a>
</div>

With the rising popularity of rock climbing, YouTube has also seen a rise in climbing-related content.
A popular video format is "guess the grade", where the YouTuber guesses the difficulty of climbs (aka boulders) featured in viewer-submitted video clips.
In the US, boulders are typically graded on the V-Scale, with easier boulders getting lower ratings (*i.e.* V0) and harder boulders getting higher rankings (the hardest being graded V17 at the time of writing).
Boulders are graded by consensus outdoors, but typically graded by the setter (the person who the boulder up) indoors.

Also rising in popularity are so-called system/training boards, which are adjustable angle walls with fixed climbing holds and LED lights to indicate which holds are usable for a given boulder.
Users can set their own boulders or access boulders set by the community through an app interface.
System boards are great because they give you access to tens of thousands of boulder at any given time, all with the footprint comparable to just a single boulder.
As a data scientist, I see a decently sized data set of boulders using a canonical set of holds---ideal for training a ML model to guess the grade!

## How It's Made
The database of boulders is obtained using the <a href="https://github.com/lemeryfertitta/BoardLib/tree/main">BoardLib library</a>, which returns an SQLite database containing all of the boulder info.
Relevant data is extracted from this database using a handful of SQL queries (focusing here on the Tension Board 1), and then pre-processed and split into training, validation, and test sets.
The Tension Board is a great playground for a few reasons: 1) there are over 70k boulders set at the time of writing, 2) the holds on the board are mirrored accross the central vertical axis of the wall, effectively doubling the number of available boulders, 3) there are a number of curated "benchmark" problems we can use as a definitive test set for each grade.

For the model itself, I'm leveraging a (small) transformer encoder consisting of 4 layers, with 2 attention heads per layer.
The holds for each climb are defined as a start hold, a finish hold, a hand hold, or a foot hold; each of these are translated into a learnable embedding.
In contrast, placement of the holds themselves on the wall are communicated to the model using a 2D sinusoidal positional encoding.
Similarly, wall angle is set using a 1D sinusoidal position encoding, which is in turn added to the global "classification" token.
The model sees one of two classification tokens---one for climbs that allow matching (putting two hands on the same hold at the same time), and another for climbs where matching is not allowed.
After passing through the transformer encoder, this encoded classification token is then fed into two fully conencted layers to predict the final V-grade of the boulder.
The model is trained by minimizing the mean square error of the predicted grade vs the given grade.
