HEADER = """
<!DOCTYPE html>
<html>
  <head>
  <style>
@keyframes spin {{
  0% {{ transform: rotate(0deg); }}
  100% {{ transform: rotate(360deg); }}
}}
</style>

    <link rel="stylesheet" href="maude-styling.css">
    <link rel="icon" type="image/x-icon" href="/favicon.ico">
    <title>{TITLE}</title>
  </head>
  <body>
  """

FOOTER = """
  </body>

  <script>
  const search_id = document.getElementById("search_id_input").value;
  const andOption = document.getElementById("andOption");
  const orOption = document.getElementById("orOption");
  const statusUpdate = document.getElementById("searchStatus");
  const extraMustContainer = document.getElementById("extraMustContainer");

  function toggleExtraMustField() {
    if (andOption.checked || orOption.checked) {
      extraMustContainer.style.display = "block";
    } else {
      extraMustContainer.style.display = "none";
    }
  }

  async function pollStatus() {
    try {
        const response = await fetch(`/maude/status/retrieve?id=${search_id}`);
        const data = await response.text();
        document.getElementById("searchStatus").textContent = data;
    } catch (err) {
      console.error("Failed to fetch status:", err);
    }
  }

  setInterval(pollStatus, 5000);

  function toggleAllYears() {
    const checkboxes = document.getElementsByName("years");
    const allChecked = Array.from(checkboxes).every(cb => cb.checked);
    checkboxes.forEach(cb => cb.checked = !allChecked);
  }
  andOption.addEventListener("change", toggleExtraMustField);
  orOption.addEventListener("change", toggleExtraMustField);

  document.getElementById("searchForm").addEventListener("submit", function () {
  document.getElementById("loadingMessage").style.display = "block";
  });
</script>



</html>
"""
BODY = """
hello
"""
SEARCH_FORM = """
<h1>MAUDE Search</h1>

    <form id="searchForm" action="/maude/search" method="GET" autocomplete="off">
    <div class="form-container">
      <div class="form-left">
      <label for="search_id">Search ID: {search_id}</label><br><br>
          <input type="hidden" id="search_id_input" name="search_id" value="{search_id}">
        <label for="name">Search:</label><br>
        <input type="text" id="query" name="query"><br><br>

        To include multiple brands, manufacturers, models, or product codes, separate search terms with a plus sign (+)

        <br><br>
        <label for="brand">Device Name/Brand:</label><br>
        <input type="text" id="brand" name="brand"><br><br>

        <label for="manu">Manufacturer:</label><br>
        <input type="text" id="manu" name="manu"><br><br>

        <label for="model_num">Model Number:</label><br>
        <input type="text" id="model_num" name="model_num"><br><br>

        <label for="code">Device Product Code:</label><br>
        <input type="text" id="code" name="code"><br><br>

        <!-- <label for="must">Must Contain:</label><br>
        <input type="text" id="must" name="must"> 
        <label for="additional">AND</label>
        <input type="radio" id="additional" name="AND">
        <label for="additional">OR</label>
        <input type="radio" id="additional" name="OR"><br> -->

<label for="must">Must Contain:</label><br>
<input type="text" id="must" name="must"><br>
<input type="radio" id="andOption" name="must_operator" value="AND">
<label for="andOption">AND</label>
<input type="radio" id="orOption" name="must_operator" value="OR">
<label for="orOption">OR</label><br><br>

<div id="extraMustContainer" style="display: none;">
  <label for="must2">Additional Must Contain:</label><br>
  <input type="text" id="must2" name="must2"><br><br>
  <!-- <input type="radio" id="andOption" name="must_operator" value="AND">
<label for="andOption">AND</label>
<input type="radio" id="orOption" name="must_operator" value="OR">
<label for="orOption">OR</label><br><br> -->
</div>



    <label>
    <input type ="checkbox" id = "search_type" name="search_type">
    Use more detailed search <br> (longer run time, requires at least one field + search filled out)
    </label> <br><br>
          <input type="submit" style="font-size: 18px;"value="Search"><br><br>

  <div id="loadingMessage" style="display: none; margin-top: 10px;">
  <span>🔄 Searching...</span>
  <span class="spinner" style="
    display: inline-block;
    width: 16px;
    height: 16px;
    border: 2px solid #ccc;
    border-top: 2px solid #333;
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
    vertical-align: middle;
    margin-left: 8px;"></span> <br><br>
  <span id="searchStatus">Submitting query...</span>
</div> 
<a href="/maude/cancel?search_id={search_id}" class="btn"> <br> Stop Search</a>
        
      </div>
      <div class="form-right">
        <p>Years included in search:</p>
        <input type="checkbox" id="2010" name = "years" value="2010" checked>
        <label for="2010">2010</label><br> 
        <input type="checkbox" id="2011" name = "years" value="2011" checked>
        <label for="2011">2011</label><br> 
        <input type="checkbox" id="2012" name = "years" value="2012" checked>
        <label for="2012">2012</label><br> 
        <input type="checkbox" id="2013" name = "years" value="2013" checked>
        <label for="2013">2013</label><br> 
        <input type="checkbox" id="2014" name = "years" value="2014" checked>
        <label for="2014">2014</label><br> 
        <input type="checkbox" id="2015" name = "years" value="2015" checked>
        <label for="2015">2015</label><br> 
        <input type="checkbox" id="2016" name = "years" value="2016" checked>
        <label for="2016">2016</label><br> 
        <input type="checkbox" id="2017" name = "years" value="2017" checked>
        <label for="2017">2017</label><br> 
        <input type="checkbox" id="2018" name = "years" value="2018" checked>
        <label for="2018">2018</label><br> 
        <input type="checkbox" id="2019" name = "years" value="2019" checked>
        <label for="2019">2019</label><br> 
        <input type="checkbox" id="2020" name = "years" value="2020" checked>
        <label for="2020">2020</label><br> 
        <input type="checkbox" id="2021" name = "years" value="2021" checked>
        <label for="2021">2021</label><br> 
        <input type="checkbox" id="2022" name = "years" value="2022" checked>
        <label for="2022">2022</label><br> 
        <input type="checkbox" id="2023" name = "years" value="2023" checked>
        <label for="2023">2023</label><br> 
        <input type="checkbox" id="2024" name = "years" value="2024" checked>
        <label for="2024">2024</label><br> 
        <input type="checkbox" id="2025" name = "years" value="2025" checked>
        <label for="2025">2025</label><br><br>

 <button id="toggleAllBtn" type="button" onclick="toggleAllYears()">Toggle All Years</button>
</div>
      </div>
    </form>
    <br>
    <br>

<div style="font-size: 18px; font-weight: bold; margin-bottom: 10px;">
  Recover Previous Search
</div>

<form id="recoverForm" action="/maude/recover" method="GET" autocomplete="off">
  <div style="margin-bottom: 15px;">
    <label for="recover" style="display: block; margin-bottom: 5px;">Enter Search ID:</label>
    <input type="text" id="recover" name="recover" style="display: block; width: 250px;">
  </div>

  <div>
    <input type="submit" value="Recover" id="sid" style="display: block; font-size: 16px; padding: 6px 16px; margin-top: 10px;">
  </div>
</form>


    """