if (!navigator.mediaDevices) {
 alert("mediaDevices is unavailable");
}

var mediaRecorder = null;
var localstream;
var status = 'hoge';

// 録音開始
rec_start = function(){
  navigator.mediaDevices.getUserMedia({audio:true})
    .then(function(stream) {
      localstream = stream;
      mediaRecorder = new MediaRecorder(stream);
      mediaRecorder.start();
    })
    .catch(function(e){
      console.log(e);
    });
}

// 録音停止
rec_stop = function(){
  mediaRecorder.stop();
  mediaRecorder.ondataavailable = function(e) {
    document.getElementById('player').src = URL.createObjectURL(e.data);
  }
  localstream.getTracks().forEach(track => track.stop());
}

record = function(){
  // ボタンの色で録音開始、停止を切り替える
  if (!mediaRecorder || status === 'hoge') {
    console.log(status);
    console.log(mediaRecorder);
    rec_start();    // 開始
    status = 'hohi';
  } else {
    console.log(status);
    rec_stop();     // 停止
    status = 'hoge';
  }
}
