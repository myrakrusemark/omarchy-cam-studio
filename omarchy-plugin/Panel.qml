import QtQuick
import Quickshell
import Quickshell.Io
import qs.Commons
import qs.Ui

// Studio Cam: icon shows the pipeline state; click for a menu.
// All actions go through the `studio-cam` CLI; state comes from `studio-cam state`.
Panel {
  id: root
  moduleName: "studio-cam"
  ipcTarget: "studio-cam"

  property string camState: "off"   // off | live | paused | loop | camoff | noobs (OBS not running)
  property int cursor: -1
  property string res: "480"   // processing size from `studio-cam res`; applies live
  property real resHoldUntil: 0  // after a click, ignore polls that started before it landed
  readonly property var resolutions: ["360", "480", "720"]

  readonly property bool on: camState !== "off"
  readonly property var actions: on
    ? (camState === "noobs" ? [{ key: "start", icon: "󰑓", label: "Restart OBS" }] : []).concat([
        { key: camState === "paused" ? "live" : "pause", icon: camState === "paused" ? "󰐊" : "󰏤",
          label: camState === "paused" ? "Resume" : "Pause (freeze frame)" },
        { key: camState === "loop" ? "live" : "loop", icon: camState === "loop" ? "󰐊" : "󰑖",
          label: camState === "loop" ? "Resume" : "Motion pause (loop last 30 s)" },
        { key: camState === "camoff" ? "live" : "camoff", icon: camState === "camoff" ? "󰕧" : "󰕨",
          label: camState === "camoff" ? "Camera on" : "Camera off" },
        { key: "quit", icon: "󰐥", label: "Quit Studio Cam" }
      ])
    : [
        { key: "start", icon: "󰕧", label: "Start Studio Cam" },
        { key: "tune", icon: "󰒓", label: "Start and tune filters" }
      ]

  readonly property string icon: camState === "noobs" ? "󰀦" : camState === "paused" ? "󰏤" : (camState === "loop" ? "󰑖" : (camState === "camoff" ? "󰕨" : (on ? "󰕧" : "󰗟")))
  readonly property string statusText: ({ off: "Off", live: "Live", paused: "Paused — frozen frame", loop: "Motion pause — looping", camoff: "Camera off",
                                             noobs: "OBS not running", starting: "Starting…" })[camState] || camState

  function refresh() { if (!stateProc.running) stateProc.running = true }

  // Start/quit take a few seconds to settle; re-check a few times after an action.
  function refreshForAWhile() {
    refreshSoon.left = 6
    refreshSoon.restart()
  }

  function run(key) {
    Quickshell.execDetached(["studio-cam", key])
    close()
    refreshForAWhile()
  }

  // Applies live, so the menu stays open to compare.
  function setRes(r) {
    if (!r) return
    Quickshell.execDetached(["studio-cam", "res", r])
    res = r
    resHoldUntil = Date.now() + 2500
  }

  // The cursor walks the actions, then the resolution buttons.
  function activate(i) {
    if (i < actions.length) run(actions[i].key)
    else setRes(resolutions[i - actions.length])
  }

  function moveCursor(dy) {
    var n = actions.length + resolutions.length
    cursor = cursor < 0 ? 0 : (cursor + dy + n) % n
  }

  onOpenedChanged: { cursor = -1; if (opened) refresh() }
  // The action list changes length with the state; an old cursor index would point elsewhere.
  onActionsChanged: cursor = -1

  implicitWidth: button.implicitWidth
  implicitHeight: button.implicitHeight

  Process {
    id: stateProc
    command: ["studio-cam", "state"]   // "<status> <resolution>"
    stdout: StdioCollector {
      waitForEnd: true
      onStreamFinished: {
        var p = text.trim().split(" ")
        if (p[0]) root.camState = p[0]
        if (p[1] && Date.now() > root.resHoldUntil) root.res = p[1]
      }
    }
  }

  Timer { interval: 2000; running: true; repeat: true; triggeredOnStart: true; onTriggered: root.refresh() }
  Timer { id: refreshSoon; interval: 700; repeat: true
    property int left: 6
    onTriggered: { root.refresh(); if (--left <= 0) stop() } }

  BarIconButton {
    id: button
    anchors.fill: parent
    bar: root.bar
    text: root.icon
    active: root.camState === "live"
    opacity: root.on ? 1 : 0.55
    tooltipText: root.opened ? "" : "Studio Cam: " + root.statusText + (root.on ? " · " + root.res + "p" : "")
    onPressed: function(b) {
      if (b === Qt.MiddleButton && root.on) root.run(root.camState === "live" ? "pause" : "live")
      else root.toggle()
    }
  }

  KeyboardPanel {
    id: panel
    anchorItem: button
    owner: root
    bar: root.bar
    open: root.opened
    focusTarget: keyCatcher
    contentWidth: panel.fittedContentWidth(Style.space(300))
    contentHeight: panel.fittedContentHeight(column.implicitHeight)

    PanelKeyCatcher {
      id: keyCatcher
      anchors.fill: parent
      onMoveRequested: function(dx, dy) { root.moveCursor(dy !== 0 ? dy : dx) }
      onActivateRequested: if (root.cursor >= 0) root.activate(root.cursor)
      onCloseRequested: root.close()
      onTabRequested: function(direction) { root.switchPanel(direction) }

      Column {
        id: column
        anchors.left: parent.left
        anchors.right: parent.right
        anchors.top: parent.top
        spacing: Style.space(12)

        Row {
          width: parent.width
          spacing: Style.space(14)

          Text {
            id: heroIcon
            text: root.icon
            color: root.bar.foreground
            font.family: root.bar.fontFamily
            font.pixelSize: Style.font.display
            anchors.verticalCenter: parent.verticalCenter
          }

          Column {
            anchors.verticalCenter: parent.verticalCenter
            spacing: Style.space(2)
            Text {
              text: "Studio Cam"
              color: root.bar.foreground
              font.family: root.bar.fontFamily
              font.pixelSize: Style.font.title
              font.bold: true
            }
            Text {
              text: (root.statusText + (root.on ? " · " + root.res + "p" : "")).toUpperCase()
              color: Qt.darker(root.bar.foreground, 1.4)
              font.family: root.bar.fontFamily
              font.pixelSize: Style.font.caption
              font.bold: true
              font.letterSpacing: 1.2
            }
          }
        }

        PanelSeparator { foreground: root.bar.foreground }

        Column {
          width: parent.width
          spacing: Style.space(6)

          Repeater {
            model: root.actions
            Button {
              required property var modelData
              required property int index
              width: parent.width
              leftAlign: true
              iconText: modelData.icon
              iconSize: Style.font.title
              text: modelData.label
              fontSize: Style.font.bodySmall
              foreground: root.bar.foreground
              fontFamily: root.bar.fontFamily
              verticalPadding: Style.spacing.controlPaddingY + Style.space(2)
              bordered: true
              hasCursor: root.cursor === index
              onClicked: root.run(modelData.key)
              onHovered: function(h) { if (h) root.cursor = index }
            }
          }
        }

        PanelSeparator { foreground: root.bar.foreground }

        Text {
          text: "RESOLUTION · LOWER IS FASTER"
          color: Qt.darker(root.bar.foreground, 1.4)
          font.family: root.bar.fontFamily
          font.pixelSize: Style.font.caption
          font.bold: true
          font.letterSpacing: 1.2
        }

        Row {
          id: resRow
          width: parent.width
          spacing: Style.space(6)

          Repeater {
            model: root.resolutions
            Button {
              required property var modelData
              required property int index
              width: (resRow.width - resRow.spacing * (root.resolutions.length - 1)) / root.resolutions.length
              text: modelData + "p"
              fontSize: Style.font.bodySmall
              foreground: root.bar.foreground
              fontFamily: root.bar.fontFamily
              verticalPadding: Style.spacing.controlPaddingY + Style.space(2)
              bordered: true
              selected: root.res === modelData
              hasCursor: root.cursor === root.actions.length + index
              onClicked: root.setRes(modelData)
              onHovered: function(h) { if (h) root.cursor = root.actions.length + index }
            }
          }
        }
      }
    }
  }
}
