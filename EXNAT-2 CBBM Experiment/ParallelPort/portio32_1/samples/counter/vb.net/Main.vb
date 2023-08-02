'   Winford Engineering
'   www.winfordeng.com
'
'   This a sample program for Winford Engineering's WECRD155B I/O Card.
'   This program uses Winford Engineering's PortIO32 Low-level library
'   Hardware setup:
'       This program assumes that you have eight LEDs connected
'       to port 1 (port A) of your I/O card.  It also assumes that you have
'       a bounceless pushbutton (or a low frequency clock source)
'       connected between Bit 0 of port 2 (port B) and ground.
'   Operation:
'       This program keeps a count of how many times the pushbutton
'       has been pressed.  The count begins at 0 and is incremented
'       every time the button is pressed.  To monitor the state of the
'       pushbutton, the program configures Port 2 to be in the input state.
'       The current value is output onto Port 1; if you have LEDs connected
'       you will be able to see the value of the counter in binary form.
'       Notice that since Port 1 is configured for output operation, you
'       should take care that you do not have any circuitry connected
'       to port 1 that could interfere with the port's outputs (See the
'       section in your manual about Data Contention).
'
Public Class Main
    Inherits System.Windows.Forms.Form

#Region " Windows Form Designer generated code "

    Public Sub New()
        MyBase.New()

        'This call is required by the Windows Form Designer.
        InitializeComponent()

        'Add any initialization after the InitializeComponent() call

    End Sub

    'Form overrides dispose to clean up the component list.
    Protected Overloads Overrides Sub Dispose(ByVal disposing As Boolean)
        If disposing Then
            If Not (components Is Nothing) Then
                components.Dispose()
            End If
        End If
        MyBase.Dispose(disposing)
    End Sub

    'Required by the Windows Form Designer
    Private components As System.ComponentModel.IContainer

    'NOTE: The following procedure is required by the Windows Form Designer
    'It can be modified using the Windows Form Designer.  
    'Do not modify it using the code editor.
    Public WithEvents cmdStart As System.Windows.Forms.Button
    Public WithEvents Label1 As System.Windows.Forms.Label
    Public WithEvents Label2 As System.Windows.Forms.Label
    Public WithEvents lblCount As System.Windows.Forms.Label
    Friend WithEvents timercheck As System.Windows.Forms.Timer
    <System.Diagnostics.DebuggerStepThrough()> Private Sub InitializeComponent()
        Me.components = New System.ComponentModel.Container()
        Me.cmdStart = New System.Windows.Forms.Button()
        Me.Label1 = New System.Windows.Forms.Label()
        Me.Label2 = New System.Windows.Forms.Label()
        Me.lblCount = New System.Windows.Forms.Label()
        Me.timercheck = New System.Windows.Forms.Timer(Me.components)
        Me.SuspendLayout()
        '
        'cmdStart
        '
        Me.cmdStart.BackColor = System.Drawing.SystemColors.Control
        Me.cmdStart.Cursor = System.Windows.Forms.Cursors.Default
        Me.cmdStart.Font = New System.Drawing.Font("Arial", 8.0!, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, CType(0, Byte))
        Me.cmdStart.ForeColor = System.Drawing.SystemColors.ControlText
        Me.cmdStart.Location = New System.Drawing.Point(64, 64)
        Me.cmdStart.Name = "cmdStart"
        Me.cmdStart.RightToLeft = System.Windows.Forms.RightToLeft.No
        Me.cmdStart.Size = New System.Drawing.Size(185, 49)
        Me.cmdStart.TabIndex = 4
        Me.cmdStart.Text = "Start "
        '
        'Label1
        '
        Me.Label1.BackColor = System.Drawing.SystemColors.Control
        Me.Label1.Cursor = System.Windows.Forms.Cursors.Default
        Me.Label1.Font = New System.Drawing.Font("Arial", 8.0!, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, CType(0, Byte))
        Me.Label1.ForeColor = System.Drawing.SystemColors.ControlText
        Me.Label1.Location = New System.Drawing.Point(48, 24)
        Me.Label1.Name = "Label1"
        Me.Label1.RightToLeft = System.Windows.Forms.RightToLeft.No
        Me.Label1.Size = New System.Drawing.Size(225, 33)
        Me.Label1.TabIndex = 7
        Me.Label1.Text = "Hit the 'Start' button to begin counting the pulses on Port 2, bit 0."
        '
        'Label2
        '
        Me.Label2.BackColor = System.Drawing.SystemColors.Control
        Me.Label2.Cursor = System.Windows.Forms.Cursors.Default
        Me.Label2.Font = New System.Drawing.Font("Arial", 8.0!, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, CType(0, Byte))
        Me.Label2.ForeColor = System.Drawing.SystemColors.ControlText
        Me.Label2.Location = New System.Drawing.Point(72, 136)
        Me.Label2.Name = "Label2"
        Me.Label2.RightToLeft = System.Windows.Forms.RightToLeft.No
        Me.Label2.Size = New System.Drawing.Size(57, 17)
        Me.Label2.TabIndex = 6
        Me.Label2.Text = "Count: "
        Me.Label2.TextAlign = System.Drawing.ContentAlignment.TopRight
        '
        'lblCount
        '
        Me.lblCount.BackColor = System.Drawing.Color.FromArgb(CType(224, Byte), CType(224, Byte), CType(224, Byte))
        Me.lblCount.Cursor = System.Windows.Forms.Cursors.Default
        Me.lblCount.Font = New System.Drawing.Font("Arial", 8.0!, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, CType(0, Byte))
        Me.lblCount.ForeColor = System.Drawing.SystemColors.ControlText
        Me.lblCount.Location = New System.Drawing.Point(136, 136)
        Me.lblCount.Name = "lblCount"
        Me.lblCount.RightToLeft = System.Windows.Forms.RightToLeft.No
        Me.lblCount.Size = New System.Drawing.Size(81, 17)
        Me.lblCount.TabIndex = 5
        '
        'timercheck
        '
        Me.timercheck.Interval = 1
        '
        'Main
        '
        Me.AutoScaleBaseSize = New System.Drawing.Size(5, 13)
        Me.ClientSize = New System.Drawing.Size(312, 213)
        Me.Controls.AddRange(New System.Windows.Forms.Control() {Me.cmdStart, Me.Label1, Me.Label2, Me.lblCount})
        Me.Name = "Main"
        Me.Text = "Counter"
        Me.ResumeLayout(False)

    End Sub

#End Region

    'Define form-level variables
    Dim BaseAddress As Short       'holds base address of CRD155B

    Private Sub cmdStart_Click(ByVal sender As System.Object, ByVal e As System.EventArgs) Handles cmdStart.Click
        timercheck.Interval = 1
        timercheck.Enabled = True

    End Sub

    Private Sub Main_Load(ByVal sender As System.Object, ByVal e As System.EventArgs) Handles MyBase.Load
        'Dimension variables
        Dim msg As String        'holds message to display to user
        Dim result As Integer       'holds the error or success codes from calls to functions from PortIO32

        'Initialize variables
        BaseAddress = 384        'assume card is at base address 384 decimal

        '*********************************************
        'First,we will output a notice and ask for user confirmation.
        'This would not normally be done in an actual application.
        msg = "Note:  This program configures Port 1 as an output port. " & _
              "Please be sure that you do not have interfering hardware " & _
              "connected to Port 1 of your card." & vbCrLf & vbCrLf & _
              "Press Ok to continue or Cancel to exit the program."
        If MsgBox(msg, MsgBoxStyle.OKCancel + MsgBoxStyle.Exclamation) = MsgBoxResult.Cancel Then
            'User hit Cancel --> End program
            End
        End If

        'Note that in this application, the error codes are checked on
        'most function calls.  In a normal application, it would not be
        'necessary to check for errors on all of the functions, but it
        'is done here to show that it is possible.

        '*********************************************
        'Configure the card.  Port A (1) should be an output and
        'Port B (2) and C (3) will be inputs.  The control byte
        'for this configuration is 139 decimal.
        result = OutByte(BaseAddress + 3, 139)
        If result = -1 Then      'error occurred
            'Display error message
            msg = "An error occurred while writing a value to a port."
            MsgBox(msg)
            'End Program
            End
        End If


    End Sub

    Private Sub timercheck_Tick(ByVal sender As System.Object, ByVal e As System.EventArgs) Handles timercheck.Tick
        'Dimension variables
        'Static variables retain their value from one execution of this function
        ' to the next.
        Static count As Integer      'counter of how many times the button has been pressed
        Static last_level As Boolean  ' Was the clock/switch line high or low last time? (True=high)
        Dim result As Integer       'holds the error or success codes from calls to functions from PortIO32
        Dim value As Integer        'holds the value read from Port B (Port 2)
        Dim level As Boolean    ' level (true=high, false=low) of just the one bit we're interested in

        result = InByte(BaseAddress + 1, value) 'read a value from Port B
        If result = -1 Then    'error occurred
            'Display error message
            MsgBox("An error occurred while reading a value from a port.")
            'stop timer from firing
            timercheck.Enabled = False
        End If

        ' Look at only bit 0 by And-ing with 1
        level = (value And 1)

        'In this program, we will assume that when the input button is pressed,
        ' it results in Bit 0 going low (0).  When the input line goes low, we
        ' increment the counter

        ' We only take action on the falling edge of the signal, that is when the
        ' level of the signal last time was high and the level this time is low.
        If (last_level = True) And (level = False) Then
            ' last time the input was high
            ' If the bit has now gone low, it's time to increment our counter
            count = count + 1      'increment counter
            lblCount.Text = count         'display current count value on the screen

            'Write current count value to the output port
            result = OutByte(BaseAddress, count)
            'Check for error during output operation
            If result = -1 Then    'Error occurred
                'Display error message
                MsgBox("An error occurred while writing a value")
                'stop timer from firing
                timercheck.Enabled = False
            End If
        End If

        ' Regardless of what level the signal was that we just read, we
        ' need to remember it for next time so we can compare to it.
        last_level = level

    End Sub

End Class
