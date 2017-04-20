function rhs = image_rhs(t,A,dummy,L,D)
rhs = (L*A).*D;
end