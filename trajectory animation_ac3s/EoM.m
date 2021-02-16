	function xdot = EoM(t,x)
%	FLIGHT Equations of Motion

%	June 12, 2015  
%	===============================================================
%	Copyright 2006-2015 by ROBERT F. STENGEL.  All rights reserved.

	global m Ixx Iyy Izz Ixz S b cBar CONHIS u tuHis deluHis uInc MODEL RUNNING
    
%   Select Aerodynamic Model

    if MODEL == 0
        AeroModel   =   @AeroModelAlpha;
    else
        AeroModel   =   @AeroModelMach;
    end

    [value,isterminal,direction] = event(t,x);
    
%	Earth-to-Body-Axis Transformation Matrix
	HEB		=	DCM(x(10),x(11),x(12));
%	Atmospheric State
    x(6)    =   min(x(6),0);    % Limit x(6) to <= 0 m
	[airDens,airPres,temp,soundSpeed]	=	Atmos(-x(6));
%	Body-Axis Wind Field
	windb	=	WindField(-x(6),x(10),x(11),x(12));
%	Body-Axis Gravity Components
	gb		=	HEB * [0;0;9.80665];

%	Air-Relative Velocity Vector
    x(1)    =   max(x(1),0);        %   Limit axial velocity to >= 0 m/s
	Va		=	[x(1);x(2);x(3)] + windb;
	V		=	sqrt(Va' * Va);
	alphar	=	atan(Va(3) / abs(Va(1)));
 %   alphar  =   min(alphar, (pi/2 - 1e-6));  %   Limit angle of attack to <= 90 deg
    
	alpha 	=	57.2957795 * alphar;
    
	betar	= 	asin(Va(2) / V);
	beta	= 	57.2957795 * betar;
	Mach	= 	V / soundSpeed;
	qbar	=	0.5 * airDens * V^2;

%	Incremental Flight Control Effects

	if CONHIS >=1 && RUNNING == 1
		[uInc]	=	interp1(tuHis,deluHis,t);
		uInc	=	(uInc)';
		uTotal	=	u + uInc;
	else
		uTotal	=	u;
    end
    
%	Force and Moment Coefficients; Thrust	
	[CD,CL,CY,Cl,Cm,Cn,Thrust]	=	AeroModel(x,uTotal,Mach,alphar,betar,V);

	qbarS	=	qbar * S;

	CX	=	-CD * cos(alphar) + CL * sin(alphar);	% Body-axis X coefficient
	CZ	= 	-CD * sin(alphar) - CL * cos(alphar);	% Body-axis Z coefficient

%	State Accelerations
   
	Xb =	(CX * qbarS + Thrust) / m;
	Yb =	CY * qbarS / m;
	Zb =	CZ * qbarS / m;
	Lb =	Cl * qbarS * b;
	Mb =	Cm * qbarS * cBar;
	Nb =	Cn * qbarS * b;
	nz	=	-Zb / 9.80665;							% Normal load factor

%	Dynamic Equations
	xd1 = Xb + gb(1) + x(9) * x(2) - x(8) * x(3);
	xd2 = Yb + gb(2) - x(9) * x(1) + x(7) * x(3);
	xd3 = Zb + gb(3) + x(8) * x(1) - x(7) * x(2);
	
	y	=	HEB' * [x(1);x(2);x(3)];
	xd4	=	y(1);
	xd5	=	y(2);
	xd6	=	y(3);
	
	xd7	= 	(Izz * Lb + Ixz * Nb - (Ixz * (Iyy - Ixx - Izz) * x(7) + ...
				(Ixz^2 + Izz * (Izz - Iyy)) * x(9)) * x(8)) / (Ixx * Izz - Ixz^2);
	xd8 = 	(Mb - (Ixx - Izz) * x(7) * x(9) - Ixz * (x(7)^2 - x(9)^2)) / Iyy;
	xd9 =	(Ixz * Lb + Ixx * Nb + (Ixz * (Iyy - Ixx - Izz) * x(9) + ...
				(Ixz^2 + Ixx * (Ixx - Iyy)) * x(7)) * x(8)) / (Ixx * Izz - Ixz^2);

	cosPitch	=	cos(x(11));
	if abs(cosPitch)	<=	0.00001
		cosPitch	=	0.00001 * sign(cosPitch);
	end
	tanPitch	=	sin(x(11)) / cosPitch;
		
	xd10	=	x(7) + (sin(x(10)) * x(8) + cos(x(10)) * x(9)) * tanPitch;
	xd11	=	cos(x(10)) * x(8) - sin(x(10)) * x(9);
	xd12	=	(sin(x(10)) * x(8) + cos(x(10)) * x(9)) / cosPitch;
	
	xdot	=	[xd1;xd2;xd3;xd4;xd5;xd6;xd7;xd8;xd9;xd10;xd11;xd12];
