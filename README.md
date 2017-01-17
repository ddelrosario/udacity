# P6 - Data Visualizations

**Summary** 
The visualization that I created is of Prosper Loan data from 2006-2014.   It shows the total volume amount of Loans borrowed by clients per state.  It is also broken down by income ranges.  From the years that I worked in the financial industry and with personal clients, I have found that most people (or even at least 90% of people) live up to their means.

**Design** 
After exploring the Prosper Loan data, I definitely wanted to investigate a by-state analysis.  As I stated, since I have worked in the financial field, I have seen many people with a lot of debt, in the form of credit cards, car loans and mortgages.  It was clear I wanted to examine loan volume and income ranges and states.

The first iteration of the map (index.html) shows Loan Volume by State for the years 2006-2014.   It uses Udacity dark blue and orange circles for Loan volume.

In the second iteration of the chart (index2.html), I took the feedback and used a softer blue and contrasted with a red for the Loan volume.   However, I agree that the first two maps don't tell a clear story.   The user doesn't know what to make of the map.

In the third iteration of the chart (index3.html), I took the feedback and added the layer of Income Range.  Using buttons on the left side, the user can click on the various income ranges and see the loan volume by state.  I believe it tells a good story of what I saw when I personally dealt with clients.  As you click through the income ranges, you can see the loan volume increase.   From the first three income ranges, under $25k, under $50k, under $75k, there is a noticeable difference in loan volumes for the states.   Then when you go from under $100k and over $100k, the difference is not so noticeable.

The saying "the more you make, the more you spend" is very true.   The map shows that for incomes under $75k and above that, the loan volume plateaus.   But even high earners are still borrowing on credit.

**Feedback**

1   Chart is plain.   The colors are plain.

2   Colors are boring.  Be careful of color-blind color schemes.

3   There is no clear story.    I don't understand what the map is supposed to be of.

**Resources** 

Data
http://eric.clst.org/Stuff/USGeoJSON
https://pypi.python.org/pypi/geopy/1.11.0
https://inkplant.com/code/state-latitudes-longitudes

Projecting
https://github.com/d3/d3-geo/blob/master/README.md#geoAlbersUsa

Reading Data
http://learnjsdata.com/read_data.html

Understanding Nesting and Showing data
http://learnjsdata.com/group_data.html
http://bl.ocks.org/phoebebright/raw/3176159/

Nest & filter
http://bl.ocks.org/jfreels/7010699

Merging States
https://gist.github.com/mbostock/5416405

Mouse Events
http://bl.ocks.org/WilliamQLiu/76ae20060e19bf42d774

Legends
http://d3-legend.susielu.com/
http://d3-legend-v3.susielu.com/

Adding Map Interaction
http://duspviz.mit.edu/d3-workshop/mapping-data-with-d3/


https://bost.ocks.org/mike/map/

Let's Make a Bubble Map
https://bost.ocks.org/mike/bubble-map/
http://www.d3noob.org/2014/02/attributes-in-d3js.html
