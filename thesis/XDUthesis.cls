%%
%% This is file `XDUthesis.cls',
%% generated with the docstrip utility.
%%
%% The original source files were:
%%
%% XDUthesis.dtx  (with options: `cls')
%% 
%% This is a generated file.
%% 
%% Copyright (C) 2015--2022 by Stick Cui <Stick_Cui@163.com>
%% 
%% This file may be distributed and/or modified under the
%% conditions of the LaTeX Project Public License, either version 1.3c
%% of this license or (at your option) any later version.
%% The latest version of this license is in:
%% 
%% http://www.latex-project.org/lppl.txt
%% 
%% and version 1.3c or later is part of all distributions of LaTeX
%% version 2008/05/04 or later.
%% 
%% To produce the documentation run the original source files ending with `.dtx'
%% through LaTeX.
%% 

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{XDUthesis}
[2019/05/11 0.1.9 Xidian University Thesis Template]
\hyphenation{XDU-Thesis}
\def\XDUthesis{X\kern -.1667em\lower .5ex\hbox {D}\kern -.125emU\textit{thesis}{}}
\def\version{0.1.9}
\LoadClass[a4paper,UTF8,zihao=-4]{ctexbook}
\RequirePackage{expl3}% LaTeX3 programming environment
\ExplSyntaxOn%
\providecommand{\g__ctex_fontset_tl}{}% platform fontset state variable
\edef\XDU@ctexfontset{\g__ctex_fontset_tl}% expanded platform fontset state variable
\ExplSyntaxOff%
\newif\if@nologo \@nologofalse %默认使用校徽Logo
\DeclareOption{nologo}{\@nologotrue}
\newif\ifXDU@windowsn \XDU@windowsnfalse % 正常windows字体，XeLaTeX下可能会乱码
\newif\ifXDU@windowsf \XDU@windowsffalse % 修正的windows字体，XeLaTeX下不能会乱码
\newif\ifXDU@mac \XDU@macfalse
\newif\ifXDU@adobe \XDU@adobefalse
\newif\ifXDU@times \XDU@timesfalse
\DeclareOption{windowsn}{\XDU@windowsntrue\XDU@timestrue}%
\DeclareOption{windowsf}{\XDU@windowsftrue\XDU@timestrue}%
\DeclareOption{mac}{\XDU@mactrue\XDU@timestrue}%
\DeclareOption{adobe}{\XDU@adobetrue\XDU@timestrue}%

\newif\ifXDU@ctex@windows \XDU@ctex@windowsfalse
\newif\ifXDU@ctex@mac \XDU@ctex@macfalse
\newif\ifXDU@ctex@adobe \XDU@ctex@adobefalse
\newif\ifXDU@ctex@fandol \XDU@ctex@fandolfalse

\newif\if@WordOneHalf \@WordOneHalffalse % 默认为LaTeX下的1.5倍行距
\DeclareOption{WordOneHalf}{\@WordOneHalftrue}
\newif\if@print \@printfalse %% 打印样式
\DeclareOption{print}{\@printtrue}
\newif\ifXDU@contentsnd \XDU@contentsndfalse
\DeclareOption{contentsnd}{\XDU@contentsndtrue}
\DeclareOption*{%
    \PassOptionsToClass{\CurrentOption}{ctexbook}%
}
\ProcessOptions\relax%
\newif\ifXDU@pdftex \XDU@pdftexfalse
\newif\ifXDU@luatex \XDU@luatexfalse
\newif\ifXDU@xetex \XDU@xetexfalse
\RequirePackage{ifxetex,ifluatex}% LaTeX engine detection
\ifxetex%
    \XDU@xetextrue
\else\ifluatex%
    \XDU@luatextrue
\else%
    \XDU@pdftextrue
\fi\fi%
\RequirePackage{etoolbox}% a toolbox of programming facilities
\newcommand{\xduifstreq}{\expandafter\ifstrequal\expandafter}%
\xduifstreq{\XDU@ctexfontset}{windows}{\XDU@ctex@windowstrue\XDU@timestrue}{%
\xduifstreq{\XDU@ctexfontset}{mac}{\XDU@ctex@mactrue\XDU@timestrue}{%
\xduifstreq{\XDU@ctexfontset}{adobe}{\XDU@ctex@adobetrue\XDU@timestrue}{%
\xduifstreq{\XDU@ctexfontset}{fandol}{\XDU@ctex@fandoltrue\XDU@timesfalse}{%
\XDU@timesfalse}}}}
\RequirePackage{ifthen}
\ifXDU@pdftex
    \RequirePackage[utf8]{inputenc}% set input encoding, document must use utf-8 encoding
    \RequirePackage[T1]{fontenc}% set font encoding to enable modern font encoding
    %- Call font package options:
    %- Text + Math: Times Roman
    \RequirePackage{newtxtext}%
    \RequirePackage[cmintegrals]{newtxmath}% load after math packages
\else% XeTeX or LuaTeX
  \ifXDU@windowsn%
    \setCJKmainfont[AutoFakeBold,ItalicFont=KaiTi]{SimSun}%
    \setCJKsansfont[AutoFakeBold]{SimHei}%
    \setCJKmonofont{FangSong}%
  \else\ifXDU@windowsf
    \setCJKmainfont[BoldFont={STZhongsong},ItalicFont={KaiTi}]{SimSun}%
    \setCJKsansfont[AutoFakeBold]{SimHei}%
    \setCJKmonofont{FangSong}%
    % \setmonofont{Courier New}
  \else\ifXDU@mac
    \setCJKmainfont[ItalicFont=Kaiti SC,BoldItalicFont=Kaiti SC Bold]{Songti SC Light}%
    \setCJKsansfont{Heiti SC}%
    \setCJKmonofont{STFangsong}%
  \else\ifXDU@adobe
    \setCJKmainfont[AutoFakeBold,ItalicFont=AdobeKaitiStd-Regular]{AdobeSongStd-Light}%
    \setCJKsansfont[AutoFakeBold]{AdobeHeitiStd-Regular}%
    \setCJKmonofont{AdobeFangsongStd-Regular}%
  \else
    \ifXDU@ctex@windows
      \setCJKmainfont[AutoFakeBold,ItalicFont=KaiTi]{SimSun}%
      \setCJKsansfont[AutoFakeBold]{SimHei}%
      \setCJKmonofont{FangSong}%
    \else\ifXDU@ctex@mac
      \setCJKmainfont[ItalicFont=Kaiti SC,BoldItalicFont=Kaiti SC Bold]{Songti SC Light}%
      \setCJKsansfont{Heiti SC}%
      \setCJKmonofont{STFangsong}%
    \else\ifXDU@ctex@adobe
      \setCJKmainfont[AutoFakeBold,ItalicFont=AdobeKaitiStd-Regular]{AdobeSongStd-Light}%
      \setCJKsansfont[AutoFakeBold]{AdobeHeitiStd-Regular}%
      \setCJKmonofont{AdobeFangsongStd-Regular}%
    \fi\fi\fi
  \fi\fi\fi\fi
  \ifXDU@times
    \setmainfont[NFSSFamily=entextrm]{Times New Roman}%
    \setsansfont[NFSSFamily=entextsf]{Times New Roman}%
  \else
    \setmainfont[NFSSFamily=entextrm]{FreeSerif}%
    \setsansfont[NFSSFamily=entextsf]{FreeSerif}%
  \fi
\fi
\RequirePackage{xcolor}
\RequirePackage{amsmath,amssymb}
\RequirePackage{graphicx}
\RequirePackage[amsmath,thmmarks,hyperref]{ntheorem}
\RequirePackage[numbers,super,square,sort&compress]{natbib}
\setlength{\bibsep}{0.2ex}%默认设置参考文献条目间距过宽，先设置为0.2ex调窄些。
\RequirePackage[toc,page,title,titletoc,header]{appendix}
\RequirePackage{listings}
\RequirePackage{CJKfntef}
\RequirePackage{longtable,multirow,hhline,tabularx,array,
                makecell,diagbox,colortbl,booktabs}
\RequirePackage[labelsep=quad]{caption}[2011/11/10]
\RequirePackage[labelformat=simple,skip=10pt]{subcaption}
\newcommand{\PreserveBackslash}[1]{\let\temp=\\#1\let\\=\temp}
\newcolumntype{C}[1]{>{\PreserveBackslash\centering}p{#1}}
\newcolumntype{R}[1]{>{\PreserveBackslash\raggedleft}p{#1}}
\newcolumntype{L}[1]{>{\PreserveBackslash\raggedright}p{#1}}
\RequirePackage[bookmarks=true,
    linkcolor=black,
    citecolor=black,
    unicode=true,
    colorlinks=true,
    pdfborder=001,
    urlcolor=black,
    bookmarksnumbered=true
]{hyperref}
\RequirePackage[chapter]{algorithm}
\RequirePackage{algpseudocode,algorithmicx}
\def\ALG@name{{算法}} %% 更改 “Algorithm” 为 “算法”，这里借鉴中国科学院大学Latex模板
\renewcommand{\thealgorithm}{\arabic{chapter}.\arabic{algorithm}} %% 算法编号跟随章编号

\if@print
    \RequirePackage[a4paper,left=4cm,right=2cm,top=3cm,bottom=2cm]{geometry}
\else
    \RequirePackage[a4paper,left=3cm,right=3cm,top=3cm,bottom=2cm]{geometry}
\fi

\RequirePackage{setspace}
\if@WordOneHalf
  \setstretch{1.62}%设置Word下的行距1.5倍
\else
  \setstretch{1.5}%设置行距1.5倍
\fi
\RequirePackage{fancyhdr}
\setlength{\headheight}{15pt}
\fancypagestyle{plain}{%为了章首页
\fancyhf{}
\fancyhead[OC]{\zihao{5}\songti\leftmark} %奇数页，章标题
\fancyhead[OR]{\zihao{-5}\thepage}
\fancyhead[EC]{\zihao{5}\songti\XDU@title} %论文题目
\fancyhead[EL]{\zihao{-5}\thepage}
\renewcommand\headrulewidth{0.75pt}
\renewcommand{\footrulewidth}{0pt}}

\fancypagestyle{myheadings}{%为了摘要
\fancyhf{}
\fancyhead[C]{\zihao{5}\songti\XDU@abstractnameheader}
\renewcommand{\headrulewidth}{0.75pt}
\renewcommand{\footrulewidth}{0pt}}
\fancypagestyle{headings}{%为了abstract
\fancyhf{}
\fancyhead[C]{\zihao{5}\songti\XDU@enabstractname}
\renewcommand{\headrulewidth}{0.75pt}
\renewcommand{\footrulewidth}{0pt}}

\def\ps@XDU@mulu{%
  \let\@oddhead\@empty%
  \let\@evenhead\@empty%
  \let\@oddfoot\@empty%
  \let\@evenfoot\@empty}

\fancypagestyle{XDU@mulu}{%为了目录
\fancyhf{}
\fancyhead[OC]{\zihao{5}\songti\leftmark}
\fancyhead[OR]{\zihao{-5}\thepage}
\fancyhead[EC]{\zihao{5}\songti\leftmark}
\fancyhead[EL]{\zihao{-5}\thepage}
\renewcommand{\headrulewidth}{0.75pt}
\renewcommand{\footrulewidth}{0pt}}
\renewcommand{\tableofcontents}{%
  \cleardoublepage%
  \chapter*{\contentsname}%
  \@mkboth{\XDU@contentsnameheader}{\XDU@contentsnameheader}%
  \@starttoc{toc}%
}%

\renewcommand\frontmatter{%
  \if@openright\cleardoublepage\else\clearpage\fi
  \@mainmatterfalse
  \pagenumbering{Roman}
  \pagestyle{XDU@mulu}}

\def\ps@XDU@main{%
  \let\@oddhead\@empty%
  \let\@evenhead\@empty%
  \let\@oddfoot\@empty%
  \let\@evenfoot\@empty}

\fancypagestyle{XDU@main}{%为了主体
\fancyhf{}
\fancyhead[OC]{\zihao{5}\songti\leftmark} %奇数页，章标题
\fancyhead[OR]{\zihao{-5}\thepage}
\fancyhead[EC]{\zihao{5}\songti\XDU@title} %论文题目
\fancyhead[EL]{\zihao{-5}\thepage}
\renewcommand\headrulewidth{0.75pt}}

\renewcommand\mainmatter{%
  \if@openright\cleardoublepage\else\clearpage\fi
  \@mainmattertrue
  \pagenumbering{arabic}
  \pagestyle{XDU@main}}
\newcommand\comtinuematter{%
  \if@openright\cleardoublepage\else\clearpage\fi
  \@mainmattertrue}


\RequirePackage{blindtext}
\@newctr{footnote}[page]
\@newctr{mpfootnote}[page]

\RequirePackage{pifont}
\renewcommand\thefootnote{\ding{\numexpr171+\value{footnote}}}
\renewcommand\thempfootnote{\ding{\numexpr171+\value{mpfootnote}}}
\ctexset{
 chapter/beforeskip = {20pt},
 chapter/format = {\zihao{3}\heiti\centering},%修正章节标题错误加粗问题
 section/format = {\zihao{4}\songti\centering},
 subsection/format = {\zihao{-4}\songti}
}
\RequirePackage{titletoc}
\ifXDU@contentsnd
  \ifXDU@pdftex
    \titlecontents{chapter}[0em]{\zihao{-4}\songti\bfseries}%
    {\thecontentslabel}%
        {}%
    {\textbf{\contentspage}}
  \else
    \titlecontents{chapter}[0em]{\zihao{-4}\bfseries}%
    {\thecontentslabel}%
        {}%
    {\textbf{\contentspage}}
  \fi
  \titlecontents{section}[3.5em]{\zihao{-4}}%
  {\contentslabel{2em}}%
      {}%
  {\textit{\contentspage}}
  \titlecontents{subsection}[6.5em]{\zihao{-4}}%
  {\contentslabel{3em}}%
      {}%
  {\textit{\contentspage}}
\else
  \ifXDU@pdftex
    \titlecontents{chapter}[0em]{\songti\bfseries}%
     {\thecontentslabel}%
        {\hspace*{0em}}%
     {\titlerule*[10bp]{\textbf{.}}\textbf{\contentspage}}
  \else
    \titlecontents{chapter}[0em]{\bfseries}%
    {\thecontentslabel}%
      {\hspace*{0em}}%
    {\titlerule*[10bp]{\textbf{.}}\textbf{\contentspage}}
  \fi
\fi

\renewcommand\theequation{\ifnum \c@chapter>\z@
  \thechapter-\fi\@arabic\c@equation}
\renewcommand{\thesubfigure}{(\alph{subfigure})}
\renewcommand{\thesubtable}{(\alph{subtable})}
\captionsetup{font={small}}%设置图表标题五号字
\renewcommand{\bibfont}{\small}%设置参考文献字体为五号字

\DeclareRobustCommand\onlinecite{\@onlinecite}
\def\@onlinecite#1{\begingroup\let\@cite\NAT@citenum\citep{#1}\endgroup}

\theorembodyfont{\rmfamily\songti}
\theoremheaderfont{\rmfamily\heiti}
\def\XDU@define@term#1{
  \expandafter\gdef\csname #1\endcsname##1{%
    \expandafter\gdef\csname XDU@#1\endcsname{##1}}
  \csname #1\endcsname{}}
\XDU@define@term{title}
\XDU@define@term{author}
\XDU@define@term{septitleA}
\XDU@define@term{septitleB}
\XDU@define@term{schoolnumber}
\XDU@define@term{school}
\XDU@define@term{major}
\XDU@define@term{class}
\XDU@define@term{supervisor}
\XDU@define@term{outersupervisor}
\XDU@define@term{thanksforname}
\XDU@define@term{thanksfornameheader}
\XDU@define@term{subject}
\XDU@define@term{abstractname}
\XDU@define@term{enabstractname}
\def\XDU@parse@keywords#1{
  \expandafter\gdef\csname XDU@#1\endcsname{}
  \expandafter\gdef\csname #1\endcsname##1{
    \@for\reserved@a:=##1\do{
      \expandafter\ifx\csname XDU@#1\endcsname\@empty\else
        \expandafter\g@addto@macro\csname XDU@#1\endcsname{
        \ignorespaces\csname XDU@#1@separator\endcsname}
      \fi
      \expandafter\expandafter\expandafter\g@addto@macro%
        \expandafter\csname XDU@#1\expandafter
        \endcsname\expandafter{\reserved@a}}}}
\XDU@parse@keywords{keywords}
\XDU@parse@keywords{enkeywords}
\newcommand{\chaptersize}[1][\zihao{3}]{#1}

\long\@xp\def\@xp\collect@@body\@xp#\@xp1\@xp\end\@xp#\@xp2\@xp{%
  \collect@@body{#1}\end{#2}}
\long\@xp\def\@xp\push@begins\@xp#\@xp1\@xp\begin\@xp#\@xp2\@xp{%
  \push@begins{#1}\begin{#2}}
\long\@xp\def\@xp\addto@envbody\@xp#\@xp1\@xp{%
  \addto@envbody{#1}}
\newcommand{\XDU@@cabstract}[1]{\long\gdef\XDU@cabstract{#1}}
\newenvironment{abstract}{\collect@body\XDU@@cabstract}{}

\newcommand{\XDU@@eabstract}[1]{\long\gdef\XDU@eabstract{#1}}
\newenvironment{enabstract}{\collect@body\XDU@@eabstract}{}

\ifXDU@pdftex%
  \ifXDU@ctex@fandol%
    \newcommand{\enheiti}{\fontfamily{FandolHei-Regular.otf}\selectfont}
  \else%
    \ifXDU@mac%
      \ifXDU@ctex@mac% 如果用户选择了Mac字体族，且ctex判断当前系统为Mac时，设置为Heiti SC，否则为SimHei，逻辑有点问题。估计不会有人用Windows系统选Mac字体的吧。
        \newcommand{\enheiti}{\fontfamily{Heiti SC}\selectfont}
      \else%
        \newcommand{\enheiti}{\fontfamily{SimHei}\selectfont}
      \fi%
    \else%
      \newcommand{\enheiti}{\fontfamily{SimHei}\selectfont}
    \fi%
  \fi%
\else%
  \ifXDU@ctex@fandol%
    \newfontfamily\enheiti{FandolHei-Regular.otf}
  \else%
    \ifXDU@mac%
      \ifXDU@ctex@mac%
        \newfontfamily\enheiti{Heiti SC}
      \else%
        \newfontfamily\enheiti{SimHei}
      \fi%
    \else%
      \newfontfamily\enheiti{SimHei} % 新加字体族，用于修正封面页标题英文字体非黑体问题，感谢lanthree <https://github.com/lanthree>同学
    \fi%
  \fi%
\fi%
\def\make@abstract{%
\cleardoublepage%
\pagenumbering{roman}
\pagestyle{myheadings}
\phantomsection
\pdfbookmark[1]{\XDU@abstractnameheader}{abscn}  %% 添加摘要页到书签
\vspace*{2em}
\centerline{\heiti\chaptersize\XDU@abstractname}\vskip5pt\par%
\vspace*{2em}
\XDU@cabstract%
\par\vspace*{2em}\noindent{\heiti\zihao{-4}\textbf{\XDU@keywordsname：\XDU@keywords}}%
\vspace*{1em}%
\cleardoublepage%
\pagestyle{headings}
\phantomsection
\pdfbookmark[1]{\XDU@enabstractname}{absen}  %% 添加Abstract页到书签
\vspace*{2em}
\centerline{\textbf{\chaptersize\XDU@enabstractname}}\vskip5pt\par%
\vspace*{2em}
\XDU@eabstract%
\par\vspace*{2em}\noindent{\textbf{\zihao{-4}
    \XDU@enkeywordsname:~\XDU@enkeywords}}%
\vspace*{1em}%
\cleardoublepage
}
\def\maketitle{
\pdfbookmark[1]{{封面}}{cover}  %% 添加封面到书签
\begin{titlepage}
%%  封面右上角班级学号格式
\begin{table}
  \raggedleft
  \renewcommand{\arraystretch}{1.3}
  \begin{tabular}{c c}
    \textbf{\zihao{-4}\XDU@classname}        & \textbf{\zihao{-4}\XDU@class}          \\ \cline{2-2}
    \textbf{\zihao{-4}\XDU@schoolnumbername} & \textbf{\zihao{-4}{\XDU@schoolnumber}} \\ \cline{2-2}
  \end{tabular}
\end{table}
\if@WordOneHalf
\if@nologo
  \vspace*{\stretch{11}}
\else
  \centering\includegraphics[width=0.5\textwidth]{./Figure/xidian.pdf}

  \vspace*{\stretch{2}}

  \begin{center}
  {\centering\heiti{\zihao{0}\XDU@subject}}
  \end{center}

  \vspace*{\stretch{3}}

  \begin{center}
  \includegraphics[width=0.25\textwidth]{./Figure/logo.pdf}
  \end{center}
\fi
\else
\if@nologo
  \vspace*{\stretch{15}}
\else
  \centering\includegraphics[width=0.5\textwidth]{./Figure/xidian.pdf}

  \vspace*{\stretch{4}}

  \begin{center}
  {\centering\heiti{\zihao{0}\XDU@subject}}
  \end{center}

  \vspace*{\stretch{5}}

  \begin{center}
  \includegraphics[width=0.25\textwidth]{./Figure/logo.pdf}
  \end{center}
\fi
\fi

\vspace*{\stretch{4}}

\begin{center}
\begin{tabular}{c C{6.5cm}}
\textbf{\zihao{3}\XDU@titlename} & {\heiti\enheiti\zihao{3}\XDU@septitleA}\\
\cline{2-2}
 & \\
 & {\heiti\enheiti\zihao{3}\XDU@septitleB}\\
\cline{2-2}
 & \\
\textbf{\zihao{3}\XDU@schoolname} & {\zihao{-3}\XDU@school}\\
\cline{2-2}
 & \\
\textbf{\zihao{3}\XDU@majorname} & {\zihao{-3}\XDU@major}\\
\cline{2-2}
 &\\
\textbf{\zihao{3}\XDU@authorname} & {\zihao{-3}\XDU@author}\\
\cline{2-2}
 &\\
\textbf{\zihao{3}\XDU@outersupervisorname} & {\zihao{-3}\XDU@outersupervisor}\\
\cline{2-2}
 &\\
\textbf{\zihao{3}\XDU@supervisorname} & {\zihao{-3}\XDU@supervisor}\\
\cline{2-2}
\cline{2-2}
\end{tabular}
\end{center}

\end{titlepage}

\pagestyle{empty}
\cleardoublepage

\begin{center}
{\textbf{\zihao{1}\XDU@declarename}}
\end{center}

\vspace*{3em}

{\songti{\zihao{4}\XDU@declaretext}}

\vspace*{8em}

{\songti\zihao{4}\XDU@authornametitle\CJKunderline{\phantom{\qquad\qquad\qquad\quad}}%
\XDU@signedname\quad\XDU@timename\@date

\XDU@supervisorhasread\CJKunderline{\phantom{\qquad\qquad\quad}}%
\XDU@signedname\quad\XDU@timename\@date}

\make@abstract%

\frontmatter
\pdfbookmark[1]{\XDU@contentsnameheader}{toc}  %% 添加目录页到书签
\tableofcontents%
\mainmatter
}
\ifxetex
  \lstset{
   showstringspaces=false,
   showspaces=false,
   tabsize=4,
   frame=lines,
   basicstyle = \XDU@codebasicfont,
   keywordstyle = \color{XDU@keywordcolor}\bfseries,
   stringstyle = \color{XDU@stringcolor}\ttfamily,
   commentstyle = \color{XDU@commentcolor}\rmfamily\itshape,
   identifierstyle=,
   columns = flexible,
   numbers = left,
   numberstyle = \footnotesize
  }
\else
  \lstset{
   showstringspaces=false,
   showspaces=false,
   tabsize=4,
   frame=lines,
   basicstyle = \XDU@codebasicfont,
   keywordstyle = \color{XDU@keywordcolor}\bfseries,
   stringstyle = \color{XDU@stringcolor}\ttfamily,
   commentstyle = \color{XDU@commentcolor}\rmfamily\itshape,
   identifierstyle=,
   columns = flexible,
   numbers = left,
   numberstyle = \footnotesize,
   extendedchars = false,
   escapechar = `
  }
\fi
\ifxetex
  \lstdefinestyle{nonumbers}
  {
   showstringspaces=false,
   showspaces=false,
   tabsize=4,
   frame=lines,
   basicstyle = \XDU@codebasicfont,
   keywordstyle = \color{XDU@keywordcolor}\bfseries,
   stringstyle = \color{XDU@stringcolor}\ttfamily,
   commentstyle = \color{XDU@commentcolor}\rmfamily\itshape,
   identifierstyle=,
   columns = flexible,
   numbers = none,
   numberstyle = \footnotesize
  }
\else
  \lstdefinestyle{nonumbers}
  {
   showstringspaces=false,
   showspaces=false,
   tabsize=4,
   frame=lines,
   basicstyle = \XDU@codebasicfont,
   keywordstyle = \color{XDU@keywordcolor}\bfseries,
   stringstyle = \color{XDU@stringcolor}\ttfamily,
   commentstyle = \color{XDU@commentcolor}\rmfamily\itshape,
   identifierstyle=,
   columns = flexible,
   numbers = none,
   numberstyle = \footnotesize,
   extendedchars = false,
   escapechar = `
  }
\fi
\newenvironment{thanksfor}{\backmatter
\chapter[\XDU@thanksfornameheader]{\XDU@thanksforname}}{\comtinuematter}
\def\XDU@setpdf@keywords{
    \hypersetup{
    pdfkeywords={\XDU@keywords}
    }
}
\AtBeginDocument{
\hypersetup{
    pdftitle={\XDU@title},
    pdfauthor={\XDU@author},%
    pdfsubject={\XDU@subject},
    pdfcreator={\XDU@author},
    pdfproducer={XDUthesis}
}
}
\AtEndOfClass{\input{XDUthesis.cfg}}
%% 
%% This package consists of the file  XDUthesis.dtx,
%%              and the derived files XDUthesis.cls,
%%                                    XDUthesis.cfg.
%% 
%%
%% End of file `XDUthesis.cls'.
