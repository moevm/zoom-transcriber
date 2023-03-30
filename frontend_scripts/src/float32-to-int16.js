export function convertFloat32ArrayToInt16Array(incomingData) {
    let l = incomingData.length;
    var buf = new Int16Array(l);
    while (l--) { var n = incomingData[l];
        var v = n < 0 ? n * 32768 : n * 32767;
        buf[l] = Math.max(-32768, Math.min(32768, v));
    }
    return buf;
};
