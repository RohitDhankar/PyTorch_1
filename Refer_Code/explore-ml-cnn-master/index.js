
window.cnn = {
  CANVAS_WIDTH: 280,
  PIXEL_WIDTH: 10,
  BLUE: '#0000FF',
  OLIVE: '#3B3C36',

  data: [],
  isDrawing: false,

  get TRANSLATED_WIDTH() {
    return this.CANVAS_WIDTH / this.PIXEL_WIDTH
  },
  get canvas() {
    return document.getElementById('canvas')
  },
  get ctx() {
    return this.canvas.getContext('2d')
  },
  getCursorPosition: function(e) {
    const {canvas} = this
    const x = e.clientX - canvas.offsetLeft
    const y = e.clientY - canvas.offsetTop

    return [x, y]
  },

  onLoad: function() {
    const {canvas} = this

    this.initData()
    canvas.onmousedown = this.onMouseDown.bind(this)
    canvas.onmousemove = this.onMouseMove.bind(this)
    document.onmouseup = this.onMouseUp.bind(this)

    this.drawGrid()
  },
  initData: function() {
    this.data = []
    for (let i = 0; i < this.TRANSLATED_WIDTH; i++) {
      const dataRow = []
      dataRow.length = this.TRANSLATED_WIDTH
      dataRow.fill(0.0)
      this.data.push(dataRow)
    }
  },
  clearRect: function() {
    this.ctx.clearRect(0, 0, this.CANVAS_WIDTH, this.CANVAS_WIDTH)
  },
  drawGrid: function() {
    const {ctx} = this

    for (
      let x = this.PIXEL_WIDTH, y = this.PIXEL_WIDTH;
      x < this.CANVAS_WIDTH;
      x += this.PIXEL_WIDTH, y += this.PIXEL_WIDTH
    ) {
      ctx.strokeStyle = this.BLUE;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, this.CANVAS_WIDTH);
      ctx.stroke();

      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(this.CANVAS_WIDTH, y);
      ctx.stroke();
    }
  },
  onMouseDown: function(e) {
    this.isDrawing = true

    const [x, y] = this.getCursorPosition(e)
    this.fillSquare(x, y)
  },
  onMouseUp: function() {
    this.isDrawing = false
  },
  onMouseMove: function(e) {
    if (!this.isDrawing) return

    const [x, y] = this.getCursorPosition(e)
    this.fillSquare(x, y)
  },
  fillSquare: function(x, y) {
    const {ctx} = this

    const xPixel = Math.floor(x / this.PIXEL_WIDTH)
    const yPixel = Math.floor(y / this.PIXEL_WIDTH)
    this.data[yPixel][xPixel] = 1

    ctx.fillStyle = this.OLIVE
    ctx.fillRect(
      xPixel * this.PIXEL_WIDTH, yPixel * this.PIXEL_WIDTH,
      this.PIXEL_WIDTH, this.PIXEL_WIDTH
    )
  },
  resetCanvas: function() {
    this.isDrawing = false
    this.initData()
    this.clearRect()
    this.clearValue()
    this.drawGrid()
  },
  clearValue: function() {
    const inputElem = document.getElementById('digit')
    inputElem.value = ''
  },
  makeLoading: function() {
    const containerElem = document.getElementById('main-container')
    const loadingElem = document.createElement('span')
    loadingElem.setAttribute('id', 'loading')
    loadingElem.textContent = 'Loading...'
    containerElem.appendChild(loadingElem)

    const inputElems = containerElem.getElementsByTagName('input')
    for (let i = 0; i < inputElems.length; i++) {
      inputElems[i].setAttribute('disabled', true)
    }
  },
  makeActive: function() {
    const containerElem = document.getElementById('main-container')
    const loadingElem = document.getElementById('loading')
    if (loadingElem) containerElem.removeChild(loadingElem)

    const inputElems = containerElem.getElementsByTagName('input')
    for (let i = 0; i < inputElems.length; i++) {
      inputElems[i].removeAttribute('disabled')
    }
  },
  train: function() {
    this.sendData({}, 'train')
  },
  test: function() {
    const {value} = document.getElementById('digit')
    if (!value || this.data.length < 0) {
      console.log('Error: no train data to send provided')
      return
    }

    const body = {
      image: this.data,
      label: Number(value),
    }

    this.sendData(body, 'test')
  },
  sendData: async function(body, path='') {
    let parsedBody
    try {
      parsedBody = JSON.stringify(body)
    } catch(e) {
      console.error('Invalid body')
      return
    }

    const headers = new Headers()
    headers.append('Content-Length', parsedBody.length.toString())
    headers.append('Connection', 'close')

    this.makeLoading()

    let isError = false
    const response = await fetch(
      `http://localhost:8000/${path}`,
      {
        method: 'POST',
        body: parsedBody,
        headers,
      }
    ).catch(error => {
      console.error(error)
      isError = true
    })
    this.makeActive()
    if (isError) return

    console.log(response)
  }
}
