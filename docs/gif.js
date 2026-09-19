
(function (root) {
  "use strict";

  /* Fixed 8 colour palette, indexes referenced by the drawing code. */
  var PALETTE = [
    [8, 8, 10],     /* 0 background      */
    [38, 38, 43],   /* 1 unsolved cell   */
    [52, 210, 123], /* 2 easy            */
    [245, 177, 61], /* 3 medium          */
    [247, 109, 109],/* 4 hard            */
    [139, 139, 149],/* 5 unrated / dim   */
    [250, 250, 250],/* 6 ink             */
    [22, 22, 26]    /* 7 panel           */
  ];
  var MIN_CODE_SIZE = 3; /* 2^3 = 8 entries */

  /** MSB-first bit packer that emits GIF sub-blocks of at most 255 bytes. */
  function BitWriter() {
    this.bytes = [];
    this.cur = 0;
    this.bits = 0;
  }
  BitWriter.prototype.write = function (code, len) {
    /* GIF packs codes least-significant-bit first within the byte stream. */
    this.cur |= code << this.bits;
    this.bits += len;
    while (this.bits >= 8) {
      this.bytes.push(this.cur & 0xff);
      this.cur >>= 8;
      this.bits -= 8;
    }
  };
  BitWriter.prototype.flush = function () {
    if (this.bits > 0) {
      this.bytes.push(this.cur & 0xff);
      this.cur = 0;
      this.bits = 0;
    }
  };

  /**
   * LZW-compress one frame of palette indexes.
   *
   * Growth rule, from giflib: codes are assigned from eoi+1 upward, and the
   * width increases once the next code to assign passes 2^width. Every code
   * actually emitted therefore still fits the width the decoder is using.
   */
  function lzw(indexes) {
    var clear = 1 << MIN_CODE_SIZE;
    var eoi = clear + 1;
    var out = new BitWriter();

    var dict, next, width;
    function reset() {
      dict = Object.create(null);
      next = eoi + 1;
      width = MIN_CODE_SIZE + 1;
    }
    reset();
    out.write(clear, width);

    var cur = indexes[0];
    for (var i = 1; i < indexes.length; i++) {
      var k = indexes[i];
      var key = cur + "_" + k;
      var found = dict[key];
      if (found !== undefined) {
        cur = found;
        continue;
      }
      out.write(cur, width);
      if (next < 4096) {
        dict[key] = next;
        next++;
        if (next > 1 << width && width < 12) width++;
      } else {
        /* Table full: tell the decoder to start over. The clear code goes out
           at the OLD width, then both sides reset to the minimum. */
        out.write(clear, width);
        reset();
      }
      cur = k;
    }
    out.write(cur, width);
    out.write(eoi, width);
    out.flush();
    return out.bytes;
  }

  function GifWriter(width, height) {
    this.w = width;
    this.h = height;
    this.parts = [];
    this.header();
  }

  GifWriter.prototype.push = function (arr) {
    this.parts.push(arr);
  };

  GifWriter.prototype.header = function () {
    var b = [];
    "GIF89a".split("").forEach(function (c) { b.push(c.charCodeAt(0)); });
    b.push(this.w & 0xff, this.w >> 8, this.h & 0xff, this.h >> 8);
    /* Global colour table, 8 entries: 0x80 | (size exponent - 1) = 0x82. */
    b.push(0x80 | 0x02, 0x00, 0x00);
    PALETTE.forEach(function (c) { b.push(c[0], c[1], c[2]); });
    /* Netscape extension: loop forever. */
    b.push(0x21, 0xff, 0x0b);
    "NETSCAPE2.0".split("").forEach(function (c) { b.push(c.charCodeAt(0)); });
    b.push(0x03, 0x01, 0x00, 0x00, 0x00);
    this.push(b);
  };

  /** delayMs is rounded to the GIF clock, which ticks every 10ms. */
  GifWriter.prototype.frame = function (indexes, delayMs) {
    var delay = Math.max(2, Math.round(delayMs / 10));
    var b = [];
    b.push(0x21, 0xf9, 0x04, 0x00, delay & 0xff, delay >> 8, 0x00, 0x00);
    b.push(0x2c, 0x00, 0x00, 0x00, 0x00, this.w & 0xff, this.w >> 8, this.h & 0xff, this.h >> 8, 0x00);
    b.push(MIN_CODE_SIZE);
    this.push(b);

    var data = lzw(indexes);
    for (var i = 0; i < data.length; i += 255) {
      var chunk = data.slice(i, i + 255);
      this.push([chunk.length].concat(chunk));
    }
    this.push([0x00]);
  };

  GifWriter.prototype.blob = function () {
    this.push([0x3b]);
    var size = this.parts.reduce(function (a, p) { return a + p.length; }, 0);
    var bytes = new Uint8Array(size);
    var at = 0;
    this.parts.forEach(function (p) {
      for (var i = 0; i < p.length; i++) bytes[at++] = p[i];
    });
    return new Blob([bytes], { type: "image/gif" });
  };

  /**
   * Map RGBA pixels onto the palette by nearest colour. The drawing code paints
   * in exact palette colours, so the only approximated pixels are the
   * antialiased edges of text, which snap to ink or background.
   */
  function quantize(rgba) {
    var out = new Uint8Array(rgba.length / 4);
    for (var i = 0, p = 0; i < rgba.length; i += 4, p++) {
      var r = rgba[i], g = rgba[i + 1], b = rgba[i + 2];
      var best = 0, bestD = Infinity;
      for (var c = 0; c < PALETTE.length; c++) {
        var dr = r - PALETTE[c][0], dg = g - PALETTE[c][1], db = b - PALETTE[c][2];
        var d = dr * dr + dg * dg + db * db;
        if (d < bestD) { bestD = d; best = c; }
      }
      out[p] = best;
    }
    return out;
  }

  root.DMLGif = {
    Writer: GifWriter,
    quantize: quantize,
    palette: PALETTE,
    css: function (i) {
      var c = PALETTE[i];
      return "rgb(" + c[0] + "," + c[1] + "," + c[2] + ")";
    }
  };
})(window);
