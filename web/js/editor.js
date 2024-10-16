function disableDrawing(){
    window.__canvas.isDrawingMode = false;
}

function enableDrawing(){
    window.__canvas.isDrawingMode = true;
    $('#drawing-mode-options').show();
}

function checkRaster(){
    log("click raster input checker");
    rasterLayerCnt++;
    var checkbox = document.getElementById('c_raster');
    var div = document.getElementById('raster');
    if (checkbox.checked == true)
    {
        div.style.display = 'block';
    }
    else{
        div.style.display = 'none';
    }
}

function checkScaledRaster(){
    var checkbox = document.getElementById('c_scaled_raster');
    if (checkbox.checked == true)
    {
        displayLayer(LAYER_RASTER, false);
    }
    else{
        displayLayer(LAYER_RASTER, true);
    }
}

function checkVecOutput(){
    log("click vector output (Raw) checker");
    rawLayerCnt++;
    var checkbox = document.getElementById('c_vec_out');
    if (checkbox.checked == true)
    {
        displayLayer(LAYER_VEC_RAW, false);
    }
    else{
        displayLayer(LAYER_VEC_RAW, true);
    }
}

function checkVecUpdate(){
    log("click vector output (Update) checker");
    updateLayerCnt++;
    var checkbox = document.getElementById('c_vec_update');
    if (checkbox.checked == true)
    {
        displayLayer(LAYER_VEC_REFINE, false);
    }
    else{
        displayLayer(LAYER_VEC_REFINE, true);
    }
}

function checkVecPost(){
    log("click vector output (Final) checker");
    finalLayerCnt++;
    var checkbox = document.getElementById('c_vec_final');
    if (checkbox.checked == true)
    {
        displayLayer(LAYER_VEC_FINAL, false);
    }
    else{
        displayLayer(LAYER_VEC_FINAL, true);
    }
}

function checkKeypoint(){
    log("click keypoint checker");
    kpLayerCnt++;
    var checkbox = document.getElementById('c_keypoints');
    if (checkbox.checked == true)
    {
        displayLayer(LAYER_KEYPOINT, false);
        if(!conciseMode){
            document.querySelector('#kp-div').style.display = "inline-block";
        }
    }
    else{
        displayLayer(LAYER_KEYPOINT, true);
        document.querySelector('#kp-div').style.display = "none";
    }
}

function checkKeypointRadius(){
    log("click Keypoint Radius checker");
    var checkbox = document.getElementById('c_kp_radius');
    if (checkbox.checked == true)
    {
        
    }
    else{
        
    }
    updateKpSize(getZoom());  // update size
}

function checkCleanUsm(){
    log("click erase checker");
    eraseCnt++;
    var checkbox = document.getElementById('c_clean_usm');
    if (checkbox.checked == true)
    {
        
    }
    else{
        
    }
}

function checkUsm(){
    log("click usm checker");
    usmLayerCnt++;
    var checkbox = document.getElementById('c_usm');
    if (checkbox.checked == true)
    {
        displayLayer(LAYER_USM, false);
        window.manager.activeLayer = window.manager.getLayer(LAYER_USM);
        updateLayerEvt(LAYER_KEYPOINT, false);
        document.querySelector('#usm-div').style.display = "inline-block";
        enableDrawing();
    }
    else{
        displayLayer(LAYER_USM, true);
        updateLayerEvt(LAYER_KEYPOINT, true);
        window.manager.activeLayer = window.manager.getLayer(LAYER_KEYPOINT);
        document.querySelector('#usm-div').style.display = "none";
        // disable drawing mode
        disableDrawing();
    }
    window.__canvas.discardActiveObject();
    window.__canvas.renderAll(); 
}

function checkUncertainUsm(){
    log("click suggested usm checker");
    unUsmLayerCnt++;
    var checkbox = document.getElementById('c_uncertain_usm');
    if (checkbox.checked == true)
    {
        displayLayer(LAYER_UNCERTAIN_USM, false);
        document.querySelector('#unusm-div').style.display = "inline-block";
    }
    else{
        displayLayer(LAYER_UNCERTAIN_USM, true);
        document.querySelector('#unusm-div').style.display = "none";
    } 
    window.__canvas.discardActiveObject();
    window.__canvas.renderAll();
}

function checkGrid(){
    log("click grid checker");
    var checkbox = document.getElementById('c_grid');
    if (checkbox.checked == true)
    {
        displayLayer(LAYER_GRID, false);
    }
    else{
        displayLayer(LAYER_GRID, true);
    }
}

function checkCanvas(){
    var checkbox = document.getElementById('c_canvas');
    var div = document.getElementById('can-bg');
    if (checkbox.checked == true)
    {
        div.style.display = 'block';
    }
    else{
        div.style.display = 'none';
    }
}

function usmRegionDisplay(){
    displayRegion = !displayRegion;
    if (displayRegion) {
        log("click Display suggested USM button");
        document.getElementById('display-label').innerHTML = 'Hide suggested USM';
        $('#display-options').show();
    }
    else {
        log("click Hide suggested USM button");
        document.getElementById('display-label').innerHTML = 'Display suggested USM';
        $('#display-options').hide();
    }
}

function hideCheckers(){
    $("#c_scaled_raster").prop('checked', false);
    $("#c_vec_out").prop('checked', false);
    $("#c_vec_update").prop('checked', false);
    $("#c_keypoints").prop('checked', false);
    $("#c_uncertain_usm").prop('checked', false);
    $("#c_usm").prop('checked', false);
    $("#c_vec_final").prop('checked', false);
    checkCheckers();
}

function checkCheckers(){
    checkVecOutput();
    checkVecUpdate();
    checkVecPost();
    checkKeypoint();
    updateKpSize(getZoom());  // update size
    updateKpOpacity();        // reset to current opacity
    checkUsm();
    checkUncertainUsm();
    checkGrid();
}

function handleUI(){
    if(conciseMode){
        document.querySelector('#kp-div').style.display = "none";
        document.querySelector('#l_grid').style.visibility = 'hidden';
        document.querySelector('#recenter').style.visibility = 'hidden';
        document.querySelector('#li-thin').style.display = "none";
        document.querySelector('#li-line').style.display = "none";
        document.querySelector('#li-bezier').style.display = "none";
        document.querySelector('#li-rdp').style.display = "none";
        document.querySelector('#li-resize').style.display = "none";
    }
    else{
        document.querySelector('#kp-div').style.display = "inline-block";
        document.querySelector('#l_grid').style.visibility = 'visible';
        document.querySelector('#recenter').style.visibility = 'visible';
        document.querySelector('#li-thin').style.display = "inline-block";
        document.querySelector('#li-line').style.display = "inline-block";
        document.querySelector('#li-bezier').style.display = "inline-block";
        document.querySelector('#li-rdp').style.display = "inline-block";
        document.querySelector('#li-resize').style.display = "inline-block";
    }
}

function overLasso(){
    if(!document.getElementById('drawing-mode').checked){
        document.getElementById("lasso-img").src="./icon/lasso.png";
    }
}

function outLasso(){
    if(!document.getElementById('drawing-mode').checked){
        document.getElementById("lasso-img").src="./icon/lasso_blue.png";
    }
}

function overPen(){
    if(!document.getElementById('pen-mode').checked){
        document.getElementById("pen-img").src="./icon/pen.png";
    }
}

function outPen(){
    if(!document.getElementById('pen-mode').checked){
        document.getElementById("pen-img").src="./icon/pen_blue.png";
    }
}

function log(content){
    let date = new Date();
    date.setMinutes(date.getMinutes() - date.getTimezoneOffset());
    if(startTime == null){
        startTime = date;
    }
    let duration = date.getTime() - startTime.getTime();
    let currentDate = '[' +  date.toISOString() + "; " + duration + ' ms] ';
    currentDate = currentDate.replace('T', ' ').replace('Z', '');
    console.log(currentDate + content)
}