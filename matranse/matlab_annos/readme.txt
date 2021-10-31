This data is released along with the following paper:
[1] "Visual Relationship Detection with Language Priors", 
Cewu Lu*, Ranjay Krishna*, Michael Bernstein, Li Fei-Fei, European Conference on Computer Vision, 
(ECCV 2016), 2016 (oral). (* = indicates equal contribution)

CITATION
===============================================================================================
If you use this data, please use the following citation:
@InProceedings{lu2016visual,
  title = {Visual Relationship Detection with Language Priors},
  author = {Lu, Cewu and Krishna, Ranjay and Bernstein, Michael and Fei-Fei, Li},
  booktitle = {European Conference on Computer Vision},
  year = {2016},
}

USING THE DATA
===============================================================================================
1. Images:
   The images used can be downloaded from:
   http://imagenet.stanford.edu/internal/jcjohns/scene_graphs/sg_dataset.zip
   We follow the same training and test splits.

2. Annotation: 
   Annotation_train.mat and annotation_test.mat contains our annotations for the 100 object categories and 70 predicates. 

   You can load and print out the data using the following matlab commands: 
   >>  load('annotation_train.mat', 'annotation_train')
   >>  annotation_train{4}   % Let's look at the 4th image in training set

   ans = 

        filename: '3008054318_8dfe57c0d0_b.jpg'
    relationship: {1x16 cell}

   annotation_train{4}.filename stores the file name of image from the zip downloaded from 1. 

   >>  annotation_train{4}.relationship{2} % the 2th relationship

       ans = 

        subBox: [120 635 109 924]
        objBox: [463 762 5 1019]
        phrase: {'truck'  'on'  'street'}

        subBox and objBox are the subject and object bounding boxes locations respectively. Our bounding box format is
        [ymin, ymax, xmin, xmax]. 
        The phrase is a list containing 3 items, the name of the subject, predicate and object.

3. Object and Predicate Categories
    The object and predicate categories are listed in 'objectListN.mat' and 'objectListN.mat'

    Let's print out all the object categories:
    >> load('objectListN.mat')
    >> for ii = 1:100; disp([num2str(ii), 'th object categroy: ',objectListN{ii}]);end
    1th object categroy: person
    2th object categroy: sky
    3th object categroy: building
    4th object categroy: truck
    5th object categroy: bus
    6th object categroy: table
    7th object categroy: shirt
    8th object categroy: chair
    9th object categroy: car
    10th object categroy: train
    11th object categroy: glasses
    12th object categroy: tree
    13th object categroy: boat
    14th object categroy: hat
    15th object categroy: trees
    16th object categroy: grass
    17th object categroy: pants
    18th object categroy: road
    19th object categroy: motorcycle
    20th object categroy: jacket
    21th object categroy: monitor
    22th object categroy: wheel
    23th object categroy: umbrella
    24th object categroy: plate
    25th object categroy: bike
    26th object categroy: clock
    27th object categroy: bag
    28th object categroy: shoe
    29th object categroy: laptop
    30th object categroy: desk
    31th object categroy: cabinet
    32th object categroy: counter
    33th object categroy: bench
    34th object categroy: shoes
    35th object categroy: tower
    36th object categroy: bottle
    37th object categroy: helmet
    38th object categroy: stove
    39th object categroy: lamp
    40th object categroy: coat
    41th object categroy: bed
    42th object categroy: dog
    43th object categroy: mountain
    44th object categroy: horse
    45th object categroy: plane
    46th object categroy: roof
    47th object categroy: skateboard
    48th object categroy: traffic light
    49th object categroy: bush
    50th object categroy: phone
    51th object categroy: airplane
    52th object categroy: sofa
    53th object categroy: cup
    54th object categroy: sink
    55th object categroy: shelf
    56th object categroy: box
    57th object categroy: van
    58th object categroy: hand
    59th object categroy: shorts
    60th object categroy: post
    61th object categroy: jeans
    62th object categroy: cat
    63th object categroy: sunglasses
    64th object categroy: bowl
    65th object categroy: computer
    66th object categroy: pillow
    67th object categroy: pizza
    68th object categroy: basket
    69th object categroy: elephant
    70th object categroy: kite
    71th object categroy: sand
    72th object categroy: keyboard
    73th object categroy: plant
    74th object categroy: can
    75th object categroy: vase
    76th object categroy: refrigerator
    77th object categroy: cart
    78th object categroy: skis
    79th object categroy: pot
    80th object categroy: surfboard
    81th object categroy: paper
    82th object categroy: mouse
    83th object categroy: trash can
    84th object categroy: cone
    85th object categroy: camera
    86th object categroy: ball
    87th object categroy: bear
    88th object categroy: giraffe
    89th object categroy: tie
    90th object categroy: luggage
    91th object categroy: faucet
    92th object categroy: hydrant
    93th object categroy: snowboard
    94th object categroy: oven
    95th object categroy: engine
    96th object categroy: watch
    97th object categroy: face
    98th object categroy: street
    99th object categroy: ramp
    100th object categroy: suitcase

    Let's print out all the predicate categories:
    >> load('predicate.mat','predicate')
    >> for ii = 1:70; disp([num2str(ii), 'th predicate categroy: ', predicate{ii}]);end
    1th predicate categroy: on
    2th predicate categroy: wear
    3th predicate categroy: has
    4th predicate categroy: next to
    5th predicate categroy: sleep next to
    6th predicate categroy: sit next to
    7th predicate categroy: stand next to
    8th predicate categroy: park next
    9th predicate categroy: walk next to
    10th predicate categroy: above
    11th predicate categroy: behind
    12th predicate categroy: stand behind
    13th predicate categroy: sit behind
    14th predicate categroy: park behind
    15th predicate categroy: in the front of
    16th predicate categroy: under
    17th predicate categroy: stand under
    18th predicate categroy: sit under
    19th predicate categroy: near
    20th predicate categroy: walk to
    21th predicate categroy: walk
    22th predicate categroy: walk past
    23th predicate categroy: in
    24th predicate categroy: below
    25th predicate categroy: beside
    26th predicate categroy: walk beside
    27th predicate categroy: over
    28th predicate categroy: hold
    29th predicate categroy: by
    30th predicate categroy: beneath
    31th predicate categroy: with
    32th predicate categroy: on the top of
    33th predicate categroy: on the left of
    34th predicate categroy: on the right of
    35th predicate categroy: sit on
    36th predicate categroy: ride
    37th predicate categroy: carry
    38th predicate categroy: look
    39th predicate categroy: stand on
    40th predicate categroy: use
    41th predicate categroy: at
    42th predicate categroy: attach to
    43th predicate categroy: cover
    44th predicate categroy: touch
    45th predicate categroy: watch
    46th predicate categroy: against
    47th predicate categroy: inside
    48th predicate categroy: adjacent to
    49th predicate categroy: across
    50th predicate categroy: contain
    51th predicate categroy: drive
    52th predicate categroy: drive on
    53th predicate categroy: taller than
    54th predicate categroy: eat
    55th predicate categroy: park on
    56th predicate categroy: lying on
    57th predicate categroy: pull
    58th predicate categroy: talk
    59th predicate categroy: lean on
    60th predicate categroy: fly
    61th predicate categroy: face
    62th predicate categroy: play with
    63th predicate categroy: sleep on
    64th predicate categroy: outside of
    65th predicate categroy: rest on
    66th predicate categroy: follow
    67th predicate categroy: hit
    68th predicate categroy: feed
    69th predicate categroy: kick
    70th predicate categroy: skate on
